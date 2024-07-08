#include "logger.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <fstream>
#include <chrono>
#include <cassert>

#include "../custosh_except.h"

namespace custosh::logging
{
    namespace
    {
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
                default:
                    assert(false);
            }
        }

        /* Auxiliary classes */
        class Logger
        {
        public:
            virtual ~Logger() = default;

            virtual void log(LogLevel level, const std::string& message) = 0;

        }; // Logger

        class ConsoleLogger : public Logger
        {
        public:
            void log(LogLevel level, const std::string& message) override
            {
                std::cout << "[" << getCurrentTime() << "] " << logLevelToString(level) << ": " << message << '\n';
            }

        }; // ConsoleLogger

        class FileLogger : public Logger
        {
        public:
            explicit FileLogger(const std::string& filename) : m_file(filename, std::ios::app)
            {
                if (!m_file.is_open()) {
                    throw CustoshException("could not open log file");
                }
            }

            void log(LogLevel level, const std::string& message) override
            {
                m_file << "[" << getCurrentTime() << "] " << logLevelToString(level) << ": " << message << '\n';
            }

        private:
            std::ofstream m_file;

        }; // FileLogger

        /* Global variables */
        std::vector<std::unique_ptr<Logger>>& getLoggers()
        {
            static std::vector<std::unique_ptr<Logger>> s_logger;
            return s_logger;
        }

    } // anonymous

    void addConsoleLogger()
    {
        getLoggers().push_back(std::make_unique<ConsoleLogger>(ConsoleLogger()));
    }

    void addFileLogger(const std::string& filename)
    {
        getLoggers().push_back(std::make_unique<FileLogger>(FileLogger(filename)));
    }

    void log(LogLevel level, const std::string& message)
    {
        for (const auto& logger: getLoggers()) {
            logger->log(level, message);
        }
    }

} // custosh::logging
