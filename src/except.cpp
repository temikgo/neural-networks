#include "except.h"

#include <exception>
#include <iostream>

namespace except {

void react() {
    try {
        throw;
    } catch (std::exception& e) {
        std::cerr << e.what() << '\n';
    } catch (...) {
        std::cerr << "Unknown exception\n";
    }
}

}  // namespace except