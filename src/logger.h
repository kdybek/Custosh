#ifndef CUSTOSH_LOGGER_H
#define CUSTOSH_LOGGER_H


#include <fstream>
#include <string>

namespace custosh
{
    enum class LogLevel {
        TRACE,
        INFO,
        ERROR,
    };

    class Logger
    {
    public:
        virtual ~Logger() = default;

        virtual void log(LogLevel level, const std::string& message) = 0;

    }; // Logger

    class ConsoleLogger : public Logger
    {
    public:
        void log(LogLevel level, const std::string& message) override;

    }; // ConsoleLogger

    class FileLogger : public Logger
    {
    public:
        explicit FileLogger(const std::string& filename);

        ~FileLogger() override;

        FileLogger(const FileLogger&) = delete;
        FileLogger& operator=(const FileLogger&) = delete;

        FileLogger(FileLogger&& other) noexcept;
        FileLogger& operator=(FileLogger&& other) noexcept;

        void log(LogLevel level, const std::string& message) override;

    private:
        std::ofstream m_file;

    }; // FileLogger

    namespace loggerManager
    {
        void addLogger(const std::shared_ptr<Logger>& consoleLogger);

        void log(LogLevel level, const std::string& message);

    } // loggerManager

} // custosh


#endif // CUSTOSH_LOGGER_H
