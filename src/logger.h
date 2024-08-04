#ifndef CUSTOSH_LOGGER_H
#define CUSTOSH_LOGGER_H


#include <fstream>
#include <string>

namespace Custosh
{
    enum class LogLevel {
        Trace,
        Info,
        Error,
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
        explicit FileLogger(std::string  filename);

        ~FileLogger() override;

        FileLogger(const FileLogger&) = delete;
        FileLogger& operator=(const FileLogger&) = delete;

        FileLogger(FileLogger&& other) noexcept;
        FileLogger& operator=(FileLogger&& other) noexcept;

        void log(LogLevel level, const std::string& message) override;

    private:
        std::string m_filename;
        std::ofstream m_file;

    }; // FileLogger

    namespace LoggerManager
    {
        void addLogger(std::unique_ptr<Logger> consoleLogger);

        void log(LogLevel level, const std::string& message);

    } // LoggerManager

} // Custosh


#endif // CUSTOSH_LOGGER_H
