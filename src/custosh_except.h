#ifndef CUSTOSH_CUSTOSH_EXCEPT_H
#define CUSTOSH_CUSTOSH_EXCEPT_H


#include <exception>
#include <string>

namespace custosh
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

} // custosh


#endif // CUSTOSH_CUSTOSH_EXCEPT_H
