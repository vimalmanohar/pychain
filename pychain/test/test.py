#!/usr/bin/env python3

import base64
import logging
import random
import tempfile
import unittest
import numpy as np
import json
import os
import shutil
from datetime import timedelta

from pychain.graph import ChainGraph, ChainGraphBatch
from pychain.loss import ChainFunction, ChainLossFunction
import simplefst
from pychain_C import set_verbose_level

import kaldi_io
import torch

from kaldi_chain import (
    gen_rand_den_fst,
    gen_rand_supervision,
    get_supervision_fsts,
    kaldi_compute_denominator,
    kaldi_compute_numerator,
    ChainSupervision,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

NUMERIC_TOL = 0.25
TOL = 1e-4
MIN_EPS = 1e-6


def test_pdf_mapping(fsts, num_pdfs):
    graphs = [ChainGraph(fst, log_domain=True, map_pdfs=True) for fst in fsts]
    num_graphs = ChainGraphBatch(
        graphs, len(fsts),
        max(t.num_transitions for t in graphs),
        max(t.num_states for t in graphs),
    )
    B = len(fsts)
    T = random.randint(10, 100)
    V = num_pdfs
    index_to_pdf = num_graphs.index_to_pdf
    U = index_to_pdf.size(1)
    assert index_to_pdf.size(0) == B

    input = torch.randn(B, T, V)

    input_indexed = torch.gather(
        input, 2, num_graphs.index_to_pdf.unsqueeze(1).expand((-1, T, -1))
    )  # B, T, U

    input_indexed_masked = input_indexed.clone()

    for b in range(B):
        for t in range(T):
            pdf_id = -2
            for u in range(U):
                if index_to_pdf[b, u] <= pdf_id:
                    assert index_to_pdf[b, u] == 0 and u > 0
                    input_indexed_masked[b, t, u] = 0
                pdf_id = index_to_pdf[b, u]
                assert input_indexed[b, t, u] == input[b, t, index_to_pdf[b, u]]

    input_reverse_indexed = torch.zeros_like(input)
    input_reverse_indexed.scatter_add_(
        2,
        index_to_pdf.unsqueeze(1).expand_as(input_indexed),
        input_indexed_masked,
    )

    for b in range(B):
        for t in range(T):
            for u in range(U):
                input_reverse_indexed[b, t, u] == input[b, t, index_to_pdf[b, u]]


def kaldi_compute_denominator_wrapper(den_fst, net_output, leaky):
    """
    This function is a wrapper around kaldi's denominator computation.
    It converts nnet_output in torch Tensor format to kaldi Matrix as
    required for kaldi's denominator computation.

    @output objf
    Denominator term of the chain objective

    @output grad
    Gradient w.r.t net_output in torch Tensor format

    @output leaky_probs
    Leaky probs computed from kaldi in torch Tensor format
    """
    B, T, N = net_output.size()
    net_output_numpy = net_output.transpose(0, 1).contiguous().view(B * T, N).detach().numpy()

    tmp_dir = tempfile.mkdtemp()
    mat_file = os.path.join(tmp_dir, "mat")
    kaldi_io.write_mat(mat_file, net_output_numpy)
    objf, leaky_probs_str = kaldi_compute_denominator(
        den_fst, N, mat_file, B, T, leaky
    )

    leaky_probs = torch.Tensor(
        [float(x) for x in leaky_probs_str.split()[1:-1]]
    )

    grad_numpy = kaldi_io.read_mat(mat_file + "_grad")
    shutil.rmtree(tmp_dir)

    assert grad_numpy.shape[0] == B * T and grad_numpy.shape[1] == N
    grad = torch.from_numpy(grad_numpy).view(T, B, N).transpose(0, 1)

    return objf, grad, leaky_probs


def kaldi_compute_numerator_wrapper(supervision, net_output):
    """
    @output objf
    Numerator term of the chain objective

    @output grad
    Gradient w.r.t net_output in torch Tensor format
    """
    B, T, N = net_output.size()
    net_output_numpy = net_output.transpose(0, 1).contiguous().view(B * T, N).detach().numpy()

    tmp_dir = tempfile.mkdtemp()
    mat_file = os.path.join(tmp_dir, "mat")
    kaldi_io.write_mat(mat_file, net_output_numpy)
    objf, = kaldi_compute_numerator(
        supervision, mat_file, B, T
    )

    grad_numpy = kaldi_io.read_mat(mat_file + "_grad")
    shutil.rmtree(tmp_dir)

    assert grad_numpy.shape[0] == B * T and grad_numpy.shape[1] == N
    grad = torch.from_numpy(grad_numpy).view(T, B, N).transpose(0, 1)

    return objf, grad


def check_grad(objf, chain_func, net_output, epsilon=1e-4, tol=NUMERIC_TOL):
    """
    This function tests gradients by numerical method
    """
    num_tries = 3
    predicted_changes = np.ones(num_tries) * 100.0
    observed_changes = np.ones(num_tries) * -100.0

    for p in range(num_tries):
        net_delta_output = torch.randn_like(net_output, device=net_output.device)
        net_delta_output.mul_(epsilon)

        predicted_changes[p] = torch.dot(
            net_output.grad.flatten(), net_delta_output.flatten()
        ).item()

        net_output_perturbed = torch.zeros_like(
            net_output, device=net_output.device, requires_grad=False
        )
        net_output_perturbed.copy_(net_output)
        net_output_perturbed.add_(net_delta_output)

        new_objf = chain_func(net_output_perturbed)

        observed_changes[p] = new_objf.item() - objf.item()

    logger.debug(f"Predicted changes: {predicted_changes}")
    logger.debug(f"Observed changes: {observed_changes}")

    correction = (predicted_changes.sum() - observed_changes.sum()) / num_tries
    observed_changes += correction

    logger.debug(
        f"Correcting observed objf changes for statistical effects, to "
        f"{observed_changes}"
    )

    rel_error = np.linalg.norm(
        predicted_changes - observed_changes
    ) / np.linalg.norm(predicted_changes)
    logger.debug(f"Relative error: {rel_error}")

    try:
        assert rel_error < tol, f"Rel error {rel_error} >= {tol} with eps {epsilon}"
    except AssertionError:
        return 1

    return 0


def check_serial(objf, net_output, input_sizes, graph_list, leaky):
    objf_serial, grad_serial = chain_function_serial(
        net_output, input_sizes, graph_list, leaky
    )
    np.testing.assert_approx_equal(
        objf,
        objf_serial,
        significant=4,
        err_msg=f"Objf mismatch batch vs serial: {objf} vs {objf_serial}",
    )

    assert np.linalg.norm(
        (grad_serial - net_output.grad).cpu()
    ) < TOL * np.linalg.norm(net_output.grad.cpu()), (
        "Grad mismatch batch vs serial"
    )


def check_cpu_vs_gpu(objf, net_output, objf_cpu, net_output_cpu):
    logger.debug("objf {}".format(objf))
    logger.debug("objf_cpu {}".format(objf_cpu))

    # CPU vs GPU
    np.testing.assert_approx_equal(
        objf_cpu,
        objf,
        significant=4,
        err_msg=f"Objf mismatch in CPU vs GPU: {objf_cpu} vs {objf}",
    )

    # Higher tolerance for mismatch in CPU vs GPU grad
    assert np.linalg.norm(
        net_output.grad.cpu() - net_output_cpu.grad
    ) < 100 * TOL * np.linalg.norm(net_output_cpu.grad), (
        "Grad mismatch in CPU vs GPU"
    )


def chain_function_serial(net_output, input_sizes, graph_list, leaky):
    """
    This function computes chain objf and gradient one sequence at a time.
    This is used to verify that batch chain computation with different
    sequence lengths is accurate.
    """
    B = net_output.shape[0]
    assert len(graph_list) == B

    net_outputs = []
    for i in range(B):
        graph_1 = ChainGraphBatch(graph_list[i], 1)
        seq_length_i = input_sizes[i].item()
        net_output_i = net_output.narrow(0, i, 1).narrow(
            1, 0, seq_length_i
        ).clone().detach().requires_grad_(True)
        net_outputs.append(net_output_i)
        objf_i, _ = ChainFunction.apply(
            net_output_i,
            input_sizes.narrow(0, i, 1),
            graph_1,
            leaky
        )
        if i == 0:
            objf = objf_i
        else:
            objf += objf_i
    objf.backward()
    grad = torch.zeros_like(net_output, requires_grad=False)
    for i in range(B):
        seq_length_i = input_sizes[i].item()
        grad.narrow(0, i, 1).narrow(1, 0, seq_length_i).copy_(net_outputs[i].grad)
    return objf, grad


def gen_random_input(N, params=None, test_kaldi=False, B=None, device='cuda'):
    if B is None:
        B = random.randint(1, 10)
    T = random.randint(10, 100)

    if params is not None:
        B, T = params["bsz"], params["max_len"]
        test_kaldi = True

    if random.randint(0, 1) == 0:
        test_kaldi = True

    # B (batch_size) x T (time) x N (net_output, N >= #pdfs in den_fst)
    zero_input = (random.randint(0, 2) == 0)
    if params is not None:
        zero_input = True

    # We want to keep values small after exp. So add -5 to all the inputs in
    # the case of 0 input.
    net_output = (
        torch.randn(B, T, N, device=device, requires_grad=True)
        if not zero_input
        else torch.zeros(B, T, N, device=device).fill_(-5).requires_grad_(True)
    )

    # input_sizes must be ints of descending order and the first
    # dimension must be equal dim(T)
    if test_kaldi:
        input_sizes = torch.tensor([T] * B).int()
    else:
        input_sizes = torch.tensor(
            [T] + sorted(
                (random.randint(min(T - 1, T // 2), T - 1) for _i in range(B - 1)),
                reverse=True
            )
        ).int()

    return net_output, input_sizes, test_kaldi


class ChainDenominatorTest(unittest.TestCase):
    def test(self):
        NUM_TRIES = 10
        num_fail = 0
        params = None
        params_json = '/tmp/pychain_test.json'
        if os.path.exists(params_json):
            NUM_TRIES = 1
            params = json.load(open(params_json))

        for _n in range(NUM_TRIES):
            try:
                if os.path.exists(params_json):
                    den_fst = simplefst.StdVectorFst.decode_to_fst(
                        base64.b64decode(params["den_fst"].encode("utf-8"))
                    )
                    N = params["num_pdfs"]
                else:
                    den_fst, N, _, _ = gen_rand_den_fst()

                graph = ChainGraph(
                    den_fst,
                    leaky_mode="hmm",
                    initial_mode="leaky",
                    final_mode="one",
                )

                # B (batch_size) x T (time) x N (net_output, N >= #pdfs in den_fst)
                net_output, input_sizes, test_kaldi = gen_random_input(N, params)
                B, T = net_output.size(0), net_output.size(1)

                den_graphs = ChainGraphBatch(graph, B)

                leaky = 1e-5 if random.randint(0, 1) == 0 else 1e-15
                if params is not None:
                    leaky = params["leaky"]

                net_output_cpu = net_output.clone().detach().cpu().requires_grad_(True)

                def chain_func(x):
                    return ChainFunction.apply(x, input_sizes, den_graphs, leaky)[0]

                objf_cpu = chain_func(net_output_cpu)
                objf_cpu.backward()

                if test_kaldi:
                    kaldi_objf, kaldi_grad, leaky_probs = kaldi_compute_denominator_wrapper(
                        den_fst, net_output_cpu, leaky
                    )
                    assert np.linalg.norm(
                        leaky_probs - graph.leaky_probs
                    ) < TOL * np.linalg.norm(leaky_probs)

                    logger.debug(f"Objf pychain vs kaldi: {objf_cpu} vs {kaldi_objf}")

                    np.testing.assert_approx_equal(
                        objf_cpu, kaldi_objf, significant=4,
                        err_msg=f"Objf mismatch pychain vs kaldi: {objf_cpu} vs {kaldi_objf}"
                    )
                    assert np.linalg.norm(
                        kaldi_grad - net_output_cpu.grad
                    ) < 100 * TOL * np.linalg.norm(net_output_cpu.grad), (
                        "Grad mismatch pychain vs kaldi; rel error = {}".format(
                            np.linalg.norm(kaldi_grad - net_output_cpu.grad)
                            / np.linalg.norm(net_output_cpu.grad)
                        )
                    )

                # Check chain func in CPU: batch vs serial
                check_serial(
                    objf_cpu, net_output_cpu, input_sizes, [graph for _ in range(B)], leaky
                )

                # Check chain func in GPU: batch vs serial
                objf = chain_func(net_output)
                objf.backward()

                check_serial(
                    objf, net_output, input_sizes, [graph for _ in range(B)], leaky
                )

                # Check grad by numerical method
                num_fail += check_grad(objf_cpu, chain_func, net_output_cpu)
                num_fail += check_grad(objf, chain_func, net_output)

                # Check CPU vs GPU
                check_cpu_vs_gpu(objf, net_output, objf_cpu, net_output_cpu)

            except AssertionError:
                if params is None:
                    with open(params_json, 'w') as fp:
                        json.dump(
                            {
                                "den_fst": base64.b64encode(
                                    simplefst.StdVectorFst.encode_fst(
                                        den_fst
                                    )
                                ).decode('utf-8'),
                                "bsz": B,
                                "max_len": T,
                                "num_pdfs": N,
                                "leaky": leaky,
                            },
                            fp=fp
                        )
                raise
        logger.info(f"Numerical failures: {num_fail} / {2 * NUM_TRIES} times")


class ChainNumeratorTest(unittest.TestCase):
    def test(self):
        NUM_TRIES = 10
        num_fail = 0
        params = None
        params_json = '/tmp/pychain_num_test.json'
        if os.path.exists(params_json):
            NUM_TRIES = 1
            params = json.load(open(params_json))

        for _n in range(NUM_TRIES):
            try:
                if os.path.exists(params_json):
                    den_fst = simplefst.StdVectorFst.decode_to_fst(
                        base64.b64decode(params["den_fst"].encode("utf-8"))
                    )
                    N = params["num_pdfs"]
                else:
                    den_fst, N, trans_model, ctx_dep = gen_rand_den_fst()

                # B (batch_size) x T (time) x N (net_output, N >= #pdfs in den_fst)
                net_output, input_sizes, test_kaldi = gen_random_input(N, params=params, test_kaldi=True, B=1)
                B, T = net_output.size(0), net_output.size(1)

                if os.path.exists(params_json):
                    supervision = ChainSupervision.decode(
                        base64.b64decode(params["supervision"].encode("utf-8"))
                    )
                else:
                    supervision = gen_rand_supervision(ctx_dep, trans_model, den_fst, T, B)

                net_output_cpu = net_output.clone().detach().cpu().requires_grad_(True)

                kaldi_objf, kaldi_grad = kaldi_compute_numerator_wrapper(supervision, net_output_cpu)

                if kaldi_objf != kaldi_objf:
                    logger.warn(f"kaldi objf is {kaldi_objf}. Skipping tests.")
                    continue

                fsts = get_supervision_fsts(supervision)
                graphs = [ChainGraph(fst) for fst in fsts]
                num_graphs = ChainGraphBatch(
                    graphs,
                    max_num_transitions=max((graph.num_transitions for graph in graphs)),
                    max_num_states=max((graph.num_states for graph in graphs))
                )

                leaky = 1e-40

                def chain_func(x):
                    return ChainFunction.apply(x, input_sizes, num_graphs, leaky)[0]

                objf_cpu = chain_func(net_output_cpu)
                objf_cpu.backward()

                logger.debug(f"Objf pychain vs kaldi: {objf_cpu} vs {kaldi_objf}")

                np.testing.assert_approx_equal(
                    objf_cpu, kaldi_objf, significant=4,
                    err_msg=f"Objf mismatch pychain vs kaldi: {objf_cpu} vs {kaldi_objf}"
                )
                assert np.linalg.norm(
                    kaldi_grad - net_output_cpu.grad
                ) < TOL * np.linalg.norm(net_output_cpu.grad), (
                    "Grad mismatch pychain vs kaldi; rel error = {}".format(
                        np.linalg.norm(kaldi_grad - net_output_cpu.grad)
                        / np.linalg.norm(net_output_cpu.grad)
                    )
                )

                # Check chain func in CPU: batch vs serial
                check_serial(
                    objf_cpu, net_output_cpu, input_sizes, graphs, leaky
                )

                # Check chain func in GPU: batch vs serial
                objf = chain_func(net_output)
                objf.backward()

                check_serial(
                    objf, net_output, input_sizes, graphs, leaky
                )

                # Check grad by numerical method
                num_fail += check_grad(objf_cpu, chain_func, net_output_cpu)
                num_fail += check_grad(objf, chain_func, net_output)

                # Check CPU vs GPU
                check_cpu_vs_gpu(objf, net_output, objf_cpu, net_output_cpu)
            except AssertionError:
                if params is None:
                    with open(params_json, 'w') as fp:
                        json.dump(
                            {
                                "den_fst": base64.b64encode(
                                    simplefst.StdVectorFst.encode_fst(
                                        den_fst
                                    )
                                ).decode('utf-8'),
                                "supervision": base64.b64encode(
                                    ChainSupervision.encode(
                                        supervision
                                    )
                                ).decode('utf-8'),
                                "bsz": B,
                                "max_len": T,
                                "num_pdfs": N,
                            },
                            fp=fp
                        )
                raise
        logger.info(f"Numerical failures: {num_fail} / {2 * NUM_TRIES} times")


class ChainNumeratorLogDomainTest(unittest.TestCase):
    def test(self):
        NUM_TRIES = 10
        num_fail = 0
        params = None
        params_json = '/tmp/pychain_num_test_logdomain.json'
        if os.path.exists(params_json):
            NUM_TRIES = 1
            params = json.load(open(params_json))

        for _n in range(NUM_TRIES):
            try:
                if os.path.exists(params_json):
                    den_fst = simplefst.StdVectorFst.decode_to_fst(
                        base64.b64decode(params["den_fst"].encode("utf-8"))
                    )
                    N = params["num_pdfs"]
                else:
                    den_fst, N, trans_model, ctx_dep = gen_rand_den_fst()

                # B (batch_size) x T (time) x N (net_output, N >= #pdfs in den_fst)
                net_output, input_sizes, test_kaldi = gen_random_input(N, params=params, test_kaldi=True)
                B, T = net_output.size(0), net_output.size(1)

                if os.path.exists(params_json):
                    supervision = ChainSupervision.decode(
                        base64.b64decode(params["supervision"].encode("utf-8"))
                    )
                else:
                    supervision = gen_rand_supervision(ctx_dep, trans_model, den_fst, T, B)


                net_output_cpu = net_output.clone().detach().cpu().requires_grad_(True)

                kaldi_objf, kaldi_grad = kaldi_compute_numerator_wrapper(supervision, net_output_cpu)

                if kaldi_objf != kaldi_objf:
                    logger.warn(f"kaldi objf is {kaldi_objf}. Skipping tests.")
                    continue

                fsts = get_supervision_fsts(supervision)

                if _n == 0:
                    test_pdf_mapping(fsts, N)

                map_pdfs = True if random.randint(0, 1) == 0 else False
                if params is not None:
                    map_pdfs = params['map_pdfs']
                graphs = [
                    ChainGraph(fst, log_domain=True, map_pdfs=map_pdfs) for fst in fsts
                ]
                num_graphs = ChainGraphBatch(
                    graphs,
                    max_num_transitions=max((graph.num_transitions for graph in graphs)),
                    max_num_states=max((graph.num_states for graph in graphs))
                )

                leaky = 1e-40

                def chain_func(x):
                    return ChainFunction.apply(x, input_sizes, num_graphs, leaky)[0]

                objf_cpu = chain_func(net_output_cpu)
                objf_cpu.backward()

                logger.debug(f"Objf pychain log-domain vs kaldi: {objf_cpu} vs {kaldi_objf}")

                np.testing.assert_approx_equal(
                    objf_cpu, kaldi_objf, significant=4,
                    err_msg=f"Objf mismatch pychain vs kaldi: {objf_cpu} vs {kaldi_objf}"
                )
                assert np.linalg.norm(
                    kaldi_grad - net_output_cpu.grad
                ) < TOL * np.linalg.norm(net_output_cpu.grad), (
                    "Grad mismatch pychain vs kaldi; rel error = {}".format(
                        np.linalg.norm(kaldi_grad - net_output_cpu.grad)
                        / np.linalg.norm(net_output_cpu.grad)
                    )
                )

                # Check chain func in CPU: batch vs serial
                check_serial(
                    objf_cpu, net_output_cpu, input_sizes, graphs, leaky
                )

                # Check grad by numerical method
                num_fail += check_grad(objf_cpu, chain_func, net_output_cpu)

            except AssertionError:
                if params is None:
                    with open(params_json, 'w') as fp:
                        json.dump(
                            {
                                "den_fst": base64.b64encode(
                                    simplefst.StdVectorFst.encode_fst(
                                        den_fst
                                    )
                                ).decode('utf-8'),
                                "supervision": base64.b64encode(
                                    ChainSupervision.encode(
                                        supervision
                                    )
                                ).decode('utf-8'),
                                "bsz": B,
                                "max_len": T,
                                "num_pdfs": N,
                                "map_pdfs": map_pdfs,
                            },
                            fp=fp
                        )
                raise
        logger.info(f"Numerical failures: {num_fail} / {NUM_TRIES} times")


class ChainLossFunctionTest(unittest.TestCase):
    def test(self):
        NUM_TRIES = 10
        num_fail = 0
        params = None
        params_json = '/tmp/pychain_loss_test.json'
        if os.path.exists(params_json):
            NUM_TRIES = 1
            params = json.load(open(params_json))

        for _n in range(NUM_TRIES):
            try:
                if os.path.exists(params_json):
                    den_fst = simplefst.StdVectorFst.decode_to_fst(
                        base64.b64decode(params["den_fst"].encode("utf-8"))
                    )
                    N = params["num_pdfs"]
                else:
                    den_fst, N, trans_model, ctx_dep = gen_rand_den_fst()

                # B (batch_size) x T (time) x N (net_output, N >= #pdfs in den_fst)
                net_output, input_sizes, test_kaldi = gen_random_input(N, params=params, test_kaldi=True)
                B, T = net_output.size(0), net_output.size(1)

                if os.path.exists(params_json):
                    supervision = ChainSupervision.decode(
                        base64.b64decode(params["supervision"].encode("utf-8"))
                    )
                else:
                    supervision = gen_rand_supervision(ctx_dep, trans_model, den_fst, T, B)


                net_output_cpu = net_output.clone().detach().cpu().requires_grad_(True)

                leaky = 1e-5 if random.randint(0, 1) == 0 else 1e-15
                if params is not None:
                    leaky = params["leaky"]

                kaldi_num_objf, kaldi_num_grad = kaldi_compute_numerator_wrapper(supervision, net_output_cpu)
                kaldi_den_objf, kaldi_den_grad, _ = kaldi_compute_denominator_wrapper(den_fst, net_output_cpu, leaky)

                kaldi_loss = -kaldi_num_objf + kaldi_den_objf
                kaldi_grad = -kaldi_num_grad + kaldi_den_grad

                if kaldi_loss != kaldi_loss:
                    logger.warn(f"kaldi loss is {kaldi_loss}. Skipping tests.")
                    continue

                fsts = get_supervision_fsts(supervision)

                if _n == 0:
                    test_pdf_mapping(fsts, N)

                map_pdfs = True if random.randint(0, 1) == 0 else False
                if params is not None:
                    map_pdfs = params['map_pdfs']
                graphs = [
                    ChainGraph(fst, log_domain=True, map_pdfs=map_pdfs) for fst in fsts
                ]
                num_graphs = ChainGraphBatch(
                    graphs,
                    max_num_transitions=max((graph.num_transitions for graph in graphs)),
                    max_num_states=max((graph.num_states for graph in graphs))
                )

                den_graph = ChainGraph(den_fst, leaky_mode="hmm", initial_mode="leaky", final_mode="one")

                def chain_loss_func(x):
                    return ChainLossFunction.apply(
                        x, input_sizes, num_graphs, den_graph, 1e-60, leaky
                    )[0]

                loss_cpu = chain_loss_func(net_output_cpu)
                loss_cpu.backward()

                logger.debug(f"Objf pychain log-domain vs kaldi: {loss_cpu} vs {kaldi_loss}")

                np.testing.assert_approx_equal(
                    loss_cpu, kaldi_loss, significant=4,
                    err_msg=f"Objf mismatch pychain vs kaldi: {loss_cpu} vs {kaldi_loss}"
                )
                assert np.linalg.norm(
                    kaldi_grad - net_output_cpu.grad
                ) < TOL * np.linalg.norm(net_output_cpu.grad), (
                    "Grad mismatch pychain vs kaldi; rel error = {}".format(
                        np.linalg.norm(kaldi_grad - net_output_cpu.grad)
                        / np.linalg.norm(net_output_cpu.grad)
                    )
                )

                # Check chain func in CPU: batch vs serial
                #check_loss_serial(
                #    loss_cpu, net_output_cpu, input_sizes, den_graph, graphs, leaky
                #)  TODO: Implement this

                # Check grad by numerical method
                num_fail += check_grad(loss_cpu, chain_loss_func, net_output_cpu)

            except AssertionError:
                if params is None:
                    with open(params_json, 'w') as fp:
                        json.dump(
                            {
                                "den_fst": base64.b64encode(
                                    simplefst.StdVectorFst.encode_fst(
                                        den_fst
                                    )
                                ).decode('utf-8'),
                                "supervision": base64.b64encode(
                                    ChainSupervision.encode(
                                        supervision
                                    )
                                ).decode('utf-8'),
                                "bsz": B,
                                "max_len": T,
                                "num_pdfs": N,
                                "map_pdfs": map_pdfs,
                            },
                            fp=fp
                        )
                raise
        logger.info(f"Numerical failures: {num_fail} / {NUM_TRIES} times")
