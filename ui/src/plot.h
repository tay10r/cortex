#pragma once

namespace cortex {

class image;

using plot_callback = void (*)(void*, const image& img);

} // namespace cortex
