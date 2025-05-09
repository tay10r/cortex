#include "viewport.h"

#include "tool.h"

#include <implot.h>

#include <GLES2/gl2.h>

namespace cortex {

namespace {

class viewport_impl final : public viewport
{
  float aspect_{ 1 };

  GLuint texture_{};

public:
  void setup() override
  {
    glGenTextures(1, &texture_);

    glBindTexture(GL_TEXTURE_2D, texture_);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  }

  void teardown() override { glDeleteTextures(1, &texture_); }

  void loop(tool* current_tool) override
  {
    if (!ImPlot::BeginPlot("##Viewport",
                           ImVec2(-1, -1),
                           ImPlotFlags_CanvasOnly | ImPlotFlags_Crosshairs | ImPlotFlags_NoFrame | ImPlotFlags_Equal)) {
      return;
    }

    ImPlot::SetupAxes("X [um]", "Y [um]");

    ImPlot::PlotImage("##Color", reinterpret_cast<ImTextureID>(texture_), ImPlotPoint(0, 0), ImPlotPoint(aspect_, 1));

    if (current_tool) {
      current_tool->render_plot();
    }

    ImPlot::EndPlot();
  }

  void update_color(const void* rgb, const int w, const int h) override
  {
    aspect_ = static_cast<float>(w) / static_cast<float>(h);

    glBindTexture(GL_TEXTURE_2D, texture_);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb);
  }
};

} // namespace

auto
viewport::create() -> std::unique_ptr<viewport>
{
  return std::make_unique<viewport_impl>();
}

} // namespace cellfi
