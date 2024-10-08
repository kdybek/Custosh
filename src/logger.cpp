#include "logger.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <chrono>
#include <cassert>
#include <utility>

#include "custosh_except.h"

namespace Custosh
{
    namespace
    {
        /* Global variables */
        std::vector<std::unique_ptr<Logger>>& getLoggers()
        {
            static std::vector<std::unique_ptr<Logger>> s_logger;
            return s_logger;
        }

        /* Auxiliary functions */
        std::string getCurrentTime()
        {
            auto now = std::chrono::system_clock::now();
            auto time = std::chrono::system_clock::to_time_t(now);
            std::ostringstream oss;
            oss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
            return oss.str();
        }

        std::string logLevelToString(LogLevel level)
        {
            switch (level) {
                case LogLevel::Trace:
                    return "TRACE";
                case LogLevel::Info:
                    return "INFO";
                case LogLevel::Error:
                    return "ERROR";
            }

            assert(false);

            // Return something to silence warnings.
            return "SOMETHING";
        }

    } // anonymous

    void ConsoleLogger::log(LogLevel level, const std::string& message)
    {
        std::cout << "[" << getCurrentTime() << "] " << logLevelToString(level) << ": " << message << '\n';
    }

    FileLogger::FileLogger(std::string filename) : m_filename(std::move(filename))
    {
    }

    FileLogger::~FileLogger()
    {
        if (m_file.is_open()) {
            m_file.close();
        }
    }

    FileLogger::FileLogger(FileLogger&& other) noexcept
            : m_filename(std::move(other.m_filename)),
              m_file(std::move(other.m_file))
    {
    }

    FileLogger& FileLogger::operator=(FileLogger&& other) noexcept
    {
        if (this != &other) {
            m_filename = std::move(other.m_filename);
            m_file = std::move(other.m_file);
        }
        return *this;
    }

    void FileLogger::log(LogLevel level, const std::string& message)
    {
        if (!m_file.is_open()) {
            m_file.open(m_filename, std::ios::trunc);
        }

        m_file << "[" << getCurrentTime() << "] " << logLevelToString(level) << ": " << message << '\n';
    }

    namespace LoggerManager
    {
        void addLogger(std::unique_ptr<Logger> consoleLogger)
        {
            getLoggers().push_back(std::move(consoleLogger));
        }

        void log(LogLevel level, const std::string& message)
        {
            for (const auto& logger: getLoggers()) {
                logger->log(level, message);
            }
        }

    } // LoggerManager

} // Custosh
