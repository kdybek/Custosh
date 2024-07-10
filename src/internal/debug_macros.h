#ifndef CUSTOSH_DEBUG_MACROS_H
#define CUSTOSH_DEBUG_MACROS_H


#ifdef CUSTOSH_DEBUG
#include <chrono>
#include <string>
#include "logger.h"
#define CUSTOSH_DEBUG_LOG_TIME(call, message) \
    do { \
        auto start = std::chrono::high_resolution_clock::now(); \
        call; \
        auto end = std::chrono::high_resolution_clock::now(); \
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); \
        std::string logMsg = message; \
        logMsg += " - "; \
        logMsg += std::to_string(elapsed.count()); \
        logMsg += "ms"; \
        LoggerManager::log(LogLevel::Trace, logMsg); \
    } while (0)
#else
#define CUSTOSH_DEBUG_LOG_TIME(call, message) call;
#endif // CUSTOSH_DEBUG


#endif // CUSTOSH_DEBUG_MACROS_H
