#pragma once

#include "../tool.h"

namespace cortex {

class focus_tool : public tool
{
public:
  static auto create(const std::vector<std::string>& args) -> std::unique_ptr<focus_tool>;

  ~focus_tool() override = default;
};

} // namespace cortex
