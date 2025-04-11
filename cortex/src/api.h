#pragma once

#include <nlohmann/json_fwd.hpp>

#include <memory>
#include <string>
#include <vector>

#include <stddef.h>

namespace cortex {

struct report_header final
{
  std::string id;

  std::string timestamp;

  std::string notes;
};

class api
{
public:
  static auto list() -> std::vector<std::string>;

  static auto create(const std::string& name) -> std::unique_ptr<api>;

  static auto create_async(const std::string& name, const size_t max_queue_size = 32) -> std::unique_ptr<api>;

  virtual ~api() = default;

  /**
   * @brief Runs the setup procedure for the API.
   * */
  virtual void setup() = 0;

  /**
   * @brief Releases the resources allocated by the API, including any threads that might have spawned.
   * */
  virtual void teardown() = 0;

  /**
   * @brief Resets the results of the API.
   * */
  virtual void reset(const report_header& header) = 0;

  /**
   * @brief Updates the results of the API with a new slide image.
   * */
  virtual void update(std::string encoded_img) = 0;

  /**
   * @brief Finalizes the results of the report, making them immutable.
   * */
  virtual void finalize() = 0;

  /**
   * @brief Gets the latest results from the report.
   * */
  [[nodiscard]] virtual auto results() -> nlohmann::json = 0;
};

} // namespace cortex
