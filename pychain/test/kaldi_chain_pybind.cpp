// Copyright      2015   Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/pybind11.h>
#include <iostream>

#include "kaldi/fstext/fstext-lib.h"
#include "kaldi/hmm/hmm-test-utils.h"
#include "kaldi/chain/chain-den-graph.h"
#include "kaldi/chain/chain-denominator.h"
#include "kaldi/chain/chain-generic-numerator.h"
#include "kaldi/chain/chain-supervision.h"
#include "kaldi/hmm/hmm-utils.h"
#include "kaldi/decoder/training-graph-compiler.h"

namespace {

namespace py = pybind11;

using kaldi::int32;
using kaldi::chain::CreateDenominatorFst;
using kaldi::ContextDependency;
using kaldi::TransitionModel;
using fst::StdVectorFst;

// computes a phone language-model FST, which has only monophone context.
void ComputeExamplePhoneLanguageModel(
    const std::vector<int32>& phones,
    StdVectorFst* g_fst) {
  g_fst->DeleteStates();
  int32 state = g_fst->AddState();
  g_fst->SetStart(state);

  kaldi::Vector<BaseFloat> probs(phones.size() + 1);
  probs.SetRandn();
  probs.ApplyPow(2.0);
  probs.Add(0.01);
  probs.Scale(1.0 / probs.Sum());

  for (size_t i = 0; i < phones.size(); i++) {
    int32 phone = phones[i];
    fst::StdArc arc(phone, phone, fst::TropicalWeight(-log(probs(i))), state);
    g_fst->AddArc(state, arc);
  }
  g_fst->SetFinal(state, fst::TropicalWeight(-log(probs(phones.size()))));
}

void ComputeExampleDenFst(
    const ContextDependency& ctx_dep,
    const TransitionModel& trans_model,
    StdVectorFst* den_graph) {
  StdVectorFst phone_lm;
  ComputeExamplePhoneLanguageModel(trans_model.GetPhones(), &phone_lm);

  CreateDenominatorFst(ctx_dep, trans_model, phone_lm, den_graph);
}

TransitionModel* GenRandTransitionModelSimple(ContextDependency** ctx_dep_out) {
  std::vector<int32> phones;
  phones.push_back(1);
  for (int32 i = 2; i < 20; i++)
    if (rand() % 2 == 0)
      phones.push_back(i);
  int32 N = 1 + rand() % 2; // context-size N is 1, 2 or 3.

  int32 P;

  // Only support monophone and left-biphone
  if (N == 1) P = 0;
  else P = 1;

  std::vector<int32> num_pdf_classes;

  ContextDependency* ctx_dep =
      kaldi::GenRandContextDependencyLarge(phones, N, P, true, &num_pdf_classes);

  kaldi::HmmTopology topo = kaldi::GenRandTopology(phones, num_pdf_classes);

  TransitionModel* trans_model = new TransitionModel(*ctx_dep, topo);

  if (ctx_dep_out == NULL)
    delete ctx_dep;
  else
    *ctx_dep_out = ctx_dep;
  return trans_model;
}

py::tuple GenRandDenFst() {
  ContextDependency* ctx_dep;
  TransitionModel* trans_model = GenRandTransitionModelSimple(&ctx_dep);
  const std::vector<int32>& phones = trans_model->GetPhones();
  StdVectorFst den_fst;
  ComputeExampleDenFst(*ctx_dep, *trans_model, &den_fst);
  int32 num_pdfs = trans_model->NumPdfs();

  return py::make_tuple(den_fst, num_pdfs, trans_model, ctx_dep);
}

StdVectorFst* GenTrivialLexiconFst(const std::vector<int32>& phones) {
  StdVectorFst* lex_fst = new StdVectorFst();

  int32 state = lex_fst->AddState();
  lex_fst->SetStart(state);
  for (const int32& phone : phones) {
    lex_fst->AddArc(
        state, fst::StdArc(phone, phone, fst::TropicalWeight::One(), state));
  }
  lex_fst->SetFinal(state, fst::TropicalWeight::One());

  fst::ArcSort(lex_fst, fst::OLabelCompare<fst::StdArc>());

  return lex_fst;
}

kaldi::chain::Supervision* GenRandSupervision(
    const ContextDependency& ctx_dep,
    const TransitionModel &trans_model,
    const StdVectorFst& den_fst,
    int num_frames,
    int num_sequences) {
  assert (num_frames > 0);

  const std::vector<int32>& phones = trans_model.GetPhones();
  StdVectorFst* lex_fst = GenTrivialLexiconFst(phones);

  std::vector<int32> disambig_syms;
  kaldi::TrainingGraphCompilerOptions opts;
  opts.self_loop_scale = 0.0;
  opts.transition_scale = 0.0;
  opts.reorder = true;

  kaldi::TrainingGraphCompiler compiler(
      trans_model, ctx_dep, lex_fst, disambig_syms, opts);
  lex_fst = NULL;

  kaldi::chain::DenominatorGraph den_graph(den_fst, trans_model.NumPdfs());
  StdVectorFst normalization_fst;
  den_graph.GetNormalizationFst(den_fst, &normalization_fst);

  std::vector<const kaldi::chain::Supervision*> sups(num_sequences, NULL);
  for (int n = 0; n < num_sequences; n++) {
    int32 phone_sequence_length = kaldi::RandInt(1, 20);
    std::vector<int32> phone_seq(phone_sequence_length);
    for (int32 i = 0; i < phone_sequence_length; i++) {
      int32 phone = phones[kaldi::RandInt(0, phones.size() - 1)];
      phone_seq[i] = phone;
    }

    StdVectorFst training_graph;

    if (!compiler.CompileGraphFromText(phone_seq, &training_graph)) {
      std::cerr << "Problem creating decoding graph for utterance";
      assert(false);
    }

    kaldi::chain::Supervision *sup = new kaldi::chain::Supervision();
    kaldi::chain::TrainingGraphToSupervisionE2e(
        training_graph, trans_model, num_frames, sup);
    assert (sup);

    // add the weight to the numerator FST so we can assert objf <= 0.
    bool ans =
        kaldi::chain::AddWeightToSupervisionFst(normalization_fst, sup);
    assert(ans);

    sups[n] = sup;
  }

  kaldi::chain::Supervision *output_supervision = new kaldi::chain::Supervision;
  kaldi::chain::MergeSupervision(sups, output_supervision);

  for (int n = 0; n < num_sequences; n++) delete sups[n];
  delete lex_fst;
  return output_supervision;
}

py::tuple KaldiComputeNumerator(
    const kaldi::chain::Supervision &supervision,
    std::string nnet_output_file,
    int num_sequences,
    int frames_per_sequence) {
  assert (supervision.num_sequences == num_sequences);
  assert (supervision.frames_per_sequence == frames_per_sequence);

  kaldi::Matrix<BaseFloat> nnet_output_cpu;
  ReadKaldiObject(nnet_output_file, &nnet_output_cpu);

  kaldi::CuMatrix<BaseFloat> nnet_output(nnet_output_cpu);
  assert(num_sequences * frames_per_sequence == nnet_output.NumRows());

  kaldi::chain::GenericNumeratorComputation numerator(supervision, nnet_output);

  BaseFloat objf;
  kaldi::CuMatrix<BaseFloat> nnet_output_deriv(
      nnet_output.NumRows(), nnet_output.NumCols());

  numerator.ForwardBackward(&objf, &nnet_output_deriv);
  kaldi::Matrix<BaseFloat> nnet_output_deriv_cpu(nnet_output_deriv);

  bool binary = false;
  kaldi::WriteKaldiObject(
      nnet_output_deriv, nnet_output_file + "_grad", binary);

  return py::make_tuple(objf);
}

py::tuple KaldiComputeDenominator(
    const StdVectorFst& den_fst,
    int32 num_pdfs,
    std::string nnet_output_file,
    int num_sequences,
    int frames_per_sequence,
    float leaky_hmm_coefficient) {
  kaldi::Matrix<BaseFloat> nnet_output_cpu;
  ReadKaldiObject(nnet_output_file, &nnet_output_cpu);

  kaldi::chain::ChainTrainingOptions opts;
  opts.leaky_hmm_coefficient = leaky_hmm_coefficient;
  opts.l2_regularize = 0;
  opts.xent_regularize = 0;

  kaldi::CuMatrix<BaseFloat> nnet_output(nnet_output_cpu);
  kaldi::chain::DenominatorGraph den_graph(den_fst, num_pdfs);
  kaldi::Vector<BaseFloat> leaky_probs(den_graph.InitialProbs());

  kaldi::chain::DenominatorComputation denominator(
      opts, den_graph, num_sequences, nnet_output);

  assert(num_sequences * frames_per_sequence == nnet_output.NumRows());
  kaldi::BaseFloat forward_prob = denominator.Forward(),
                   per_frame =
                       forward_prob / (num_sequences * frames_per_sequence);

  kaldi::CuMatrix<BaseFloat> nnet_output_deriv(
      nnet_output.NumRows(), nnet_output.NumCols());

  denominator.Backward(1.0, &nnet_output_deriv);

  kaldi::Matrix<BaseFloat> nnet_output_deriv_cpu(nnet_output_deriv);

  bool binary = false;
  kaldi::WriteKaldiObject(
      nnet_output_deriv, nnet_output_file + "_grad", binary);

  std::stringstream oss;
  leaky_probs.Write(oss, false);
  oss.str();

  return py::make_tuple(forward_prob, oss.str());
}

py::list GetSupervisionFsts(const kaldi::chain::Supervision &sup) {
  py::list fsts;

  for (auto &fst: sup.e2e_fsts) {
    fsts.append(fst);
  }

  return fsts;
}

py::bytes EncodeSupervision(kaldi::chain::Supervision &sup) {
  std::ostringstream oss;
  sup.Write(oss, true);
  return py::bytes(oss.str());
}

kaldi::chain::Supervision* DecodeToSupervision(const py::bytes &data) {
  std::string str(data);
  std::istringstream iss(str);

  kaldi::chain::Supervision *sup = new kaldi::chain::Supervision;
  sup->Read(iss, true);

  return sup;
}

PYBIND11_MODULE(kaldi_chain, m) {
  m.def("gen_rand_den_fst", &GenRandDenFst);
  m.def("gen_rand_supervision", &GenRandSupervision);
  m.def("kaldi_compute_denominator", &KaldiComputeDenominator);
  m.def("kaldi_compute_numerator", &KaldiComputeNumerator);
  m.def("get_supervision_fsts", &GetSupervisionFsts);

  py::class_<kaldi::TransitionModel>(m, "TransitionModel");
  py::class_<kaldi::chain::Supervision>(m, "ChainSupervision")
    .def_static("encode", &EncodeSupervision)
    .def_static("decode", &DecodeToSupervision);
  py::class_<kaldi::ContextDependency>(m, "ContextDependency");
}

} // namespace
