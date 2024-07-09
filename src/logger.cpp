#include "logger.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <chrono>
#include <cassert>

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
                case LogLevel::TRACE:
                    return "TRACE";
                case LogLevel::INFO:
                    return "INFO";
                case LogLevel::ERROR:
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

    FileLogger::FileLogger(const std::string& filename) : m_file(filename, std::ios::app)
    {
        if (!m_file.is_open()) {
            throw CustoshException("could not open log file");
        }
    }

    FileLogger::~FileLogger()
    {
        if (m_file.is_open()) {
            m_file.close();
        }
    }

    FileLogger::FileLogger(FileLogger&& other) noexcept: m_file(std::move(other.m_file))
    {
    }

    FileLogger& FileLogger::operator=(FileLogger&& other) noexcept
    {
        if (this != &other) {
            m_file = std::move(other.m_file);
        }
        return *this;
    }

    void FileLogger::log(LogLevel level, const std::string& message)
    {
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
