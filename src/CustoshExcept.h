#ifndef CUSTOSH_CUSTOSHEXCEPT_H
#define CUSTOSH_CUSTOSHEXCEPT_H


#include <exception>
#include <string>

namespace Custosh
{
    class CustoshException : public std::exception
    {
    private:
        std::string m_errorMessage;

    public:
        explicit CustoshException(std::string message) : m_errorMessage(std::move(message))
        {
        }

        ~CustoshException() noexcept override = default;

        [[nodiscard]] const char* what() const noexcept override
        {
            return m_errorMessage.c_str();
        }

    }; // CustoshException

} // Custosh


#endif // CUSTOSH_CUSTOSHEXCEPT_H
