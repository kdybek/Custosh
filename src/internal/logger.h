#ifndef CUSTOSH_LOGGER_H
#define CUSTOSH_LOGGER_H


#include <string>

namespace custosh::logging
{
    enum class LogLevel {
        TRACE,
        INFO,
        ERROR,
    };

    void addConsoleLogger();

    void addFileLogger(const std::string& filename);

    void log(LogLevel level, const std::string& message);

} // custosh::logging


#endif // CUSTOSH_LOGGER_H
