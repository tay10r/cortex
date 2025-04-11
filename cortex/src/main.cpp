#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>

#include <stdexcept>

#include "api.h"
#include "wbc_differential_api.h"

// impl
#include "wbc_classifier.h"
#include "wbc_localizer.h"

#include <opencv2/imgcodecs.hpp>

namespace {

namespace py = pybind11;

void
def_impl_module(py::module_ m)
{
  py::enum_<cortex::wbc_kind>(m, "WBCKind")
    .value("UNKNOWN", cortex::wbc_kind::unknown)
    .value("NETROPHIL", cortex::wbc_kind::neutrophil)
    .value("LYMPHOCYTE", cortex::wbc_kind::lymphocyte)
    .value("MONOCYTE", cortex::wbc_kind::monocyte)
    .value("EOSINOPHILS", cortex::wbc_kind::eosinophils)
    .value("BASOPHILS", cortex::wbc_kind::basophils);

  py::class_<cortex::wbc_classifier>(m, "WBCClassifier")
    .def(py::init<>())
    .def("load_model", &cortex::wbc_classifier::load_model, py::arg("model_filename"))
    .def("classify",
         [](cortex::wbc_classifier& classifier, py::array_t<uint8_t, py::array::forcecast | py::array::c_style>& img) {
           if (img.ndim() != 3) {
             throw std::runtime_error("WBC classifier expects a 3 dimensional array");
           }
           return classifier.classify(img.mutable_data(0, 0, 0), img.shape()[1], img.shape()[0]);
         });

  py::class_<cortex::wbc_localizer>(m, "WBCLocalizer")
    .def(py::init<>())
    .def("load_model", &cortex::wbc_localizer::load_model, py::arg("model_filename"))
    .def("segment_tiles",
         [](cortex::wbc_localizer& localizer, py::array_t<uint8_t, py::array::forcecast | py::array::c_style>& img) {
           if (img.ndim() != 3) {
             throw std::runtime_error("WBC classifier expects a 3 dimensional array");
           }
           auto mask = localizer.segment_tiles(img.mutable_data(0, 0, 0), img.shape()[1], img.shape()[0]);
           std::vector<ssize_t> shape{ mask.size().height, mask.size().width };
           auto result = py::array_t < uint8_t, py::array::forcecast | py::array::c_style > (shape);
           for (size_t y = 0; y < shape[0]; y++) {
             for (size_t x = 0; x < shape[1]; x++) {
               const auto p = static_cast<int>(mask.at<float>(y, x) * 255);
               result.mutable_at(y, x) = static_cast<uint8_t>(p);
             }
           }
           return result;
         })
    .def("segment",
         [](cortex::wbc_localizer& localizer, py::array_t<uint8_t, py::array::forcecast | py::array::c_style>& img) {
           if (img.ndim() != 3) {
             throw std::runtime_error("WBC classifier expects a 3 dimensional array");
           }
           auto mask = localizer.segment(img.mutable_data(0, 0, 0), img.shape()[1], img.shape()[0]);
           std::vector<ssize_t> shape{ mask.size().height, mask.size().width };
           auto result = py::array_t < uint8_t, py::array::forcecast | py::array::c_style > (shape);
           for (size_t y = 0; y < shape[0]; y++) {
             for (size_t x = 0; x < shape[1]; x++) {
               const auto p = static_cast<int>(mask.at<float>(y, x) * 255);
               result.mutable_at(y, x) = static_cast<uint8_t>(p);
             }
           }
           return result;
         });
}

} // namespace

PYBIND11_MODULE(cortex, m)
{
  py::class_<cortex::report_header>(m, "ReportHeader")
    .def(py::init<>())
    .def_readwrite("id_", &cortex::report_header::id)
    .def_readwrite("notes", &cortex::report_header::notes)
    .def_readwrite("timestamp", &cortex::report_header::timestamp);

  py::class_<cortex::api>(m, "API")
    .def_static("create", &cortex::api::create, py::arg("name"))
    .def_static("create_async", &cortex::api::create_async, py::arg("name"), py::arg("max_queue_size"))
    .def("setup", &cortex::api::setup)
    .def("teardown", &cortex::api::teardown)
    .def("reset", &cortex::api::reset, py::arg("header"))
    .def(
      "update",
      [](cortex::api& api, std::string encoded_img) { api.update(std::move(encoded_img)); },
      py::arg("encoded_img"))
    .def("finalize", &cortex::api::finalize)
    .def("results", [](cortex::api& api) -> std::string { return api.results().dump(); });

  py::class_<cortex::wbc_differential_api, cortex::api>(m, "WBCDifferentialAPI").def(py::init<>());
  def_impl_module(m.def_submodule("impl", "The API for some of the internal workings of the library."));
}
