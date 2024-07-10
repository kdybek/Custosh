#ifndef CUSTOSH_DEBUG_MACROS_H
#define CUSTOSH_DEBUG_MACROS_H


#ifdef CUSTOSH_DEBUG

#include <chrono>
#include <string>
#include "logger.h"

#define CUSTOSH_DEBUG_CALL(call) \
    do{ \
        call; \
    } while (0)

#define CUSTOSH_DEBUG_TRACE(message) \
    do { \
        LoggerManager::log(LogLevel::Trace, message); \
    } while (0)

#define CUSTOSH_DEBUG_INFO(message) \
    do { \
        LoggerManager::log(LogLevel::Info, message); \
    } while (0)

#define CUSTOSH_DEBUG_ERROR(message) \
    do { \
        LoggerManager::log(LogLevel::Error, message); \
    } while (0)

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
        CUSTOSH_DEBUG_INFO(logMsg); \
    } while (0)
#else

#define CUSTOSH_DEBUG_CALL(call)

#define CUSTOSH_DEBUG_TRACE(message)

#define CUSTOSH_DEBUG_INFO(message)

#define CUSTOSH_DEBUG_ERROR(message)

#define CUSTOSH_DEBUG_LOG_TIME(call, message) call;

#endif // CUSTOSH_DEBUG


#endif // CUSTOSH_DEBUG_MACROS_H
