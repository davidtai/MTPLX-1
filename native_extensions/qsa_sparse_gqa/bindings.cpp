// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "sparse_gqa/qsa_sparse_gqa.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "MTPLX native QSA direct-index sparse-GQA attention";

  m.def("qsa_sparse_gqa_attention", &mtplx_native::qsa_sparse_gqa_attention,
        "queries"_a, "keys"_a, "values"_a, "selected_blocks"_a, "scale"_a,
        "q_offset"_a, "key_length"_a = -1, "key_tile"_a = 128,
        "dimension_tile"_a = 32, nb::kw_only(), "stream"_a = nb::none(),
        "Direct-index sparse GQA attention over chronological QSA block ids.");

  m.def("qsa_sparse_gqa_unsupported_reason",
        &mtplx_native::qsa_sparse_gqa_unsupported_reason, "queries"_a, "keys"_a,
        "values"_a, "selected_blocks"_a, "scale"_a, "q_offset"_a,
        "key_length"_a = -1, "key_tile"_a = 128, "dimension_tile"_a = 32,
        nb::kw_only(), "stream"_a = nb::none(),
        "Empty string when the call is on contract; otherwise the reason.");
}
