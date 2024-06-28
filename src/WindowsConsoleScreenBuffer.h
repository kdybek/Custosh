#ifndef CUSTOSH_WINDOWSCONSOLESCREENBUFFER_H
#define CUSTOSH_WINDOWSCONSOLESCREENBUFFER_H


#define NOMINMAX
#include <windows.h>

#include "CustoshExcept.h"
#include "Utility.h"

namespace Custosh
{

    class WindowsConsoleScreenBuffer
    {
    public:
        WindowsConsoleScreenBuffer() : m_handle(CreateConsoleScreenBuffer(
                GENERIC_READ | GENERIC_WRITE,   // Access rights
                0,                              // No sharing
                nullptr,                        // Default security attributes
                CONSOLE_TEXTMODE_BUFFER,        // Text mode buffer
                nullptr                         // Default buffer data
        ))
        {
            if (m_handle == INVALID_HANDLE_VALUE) {
                throw CustoshException("error creating screen buffer");
            }
        }

        ~WindowsConsoleScreenBuffer()
        {
            CloseHandle(m_handle);
        }

        WindowsConsoleScreenBuffer(const WindowsConsoleScreenBuffer&) = delete;

        WindowsConsoleScreenBuffer& operator=(const WindowsConsoleScreenBuffer&) = delete;

        void draw(const BrightnessMap& bm)
        {
            for (unsigned int i = 0; i < bm.getNRows(); ++i) {
                std::string terminalOutput = bm.rowToString(i);

                DWORD charsWritten;
                COORD coord = {0, static_cast<SHORT>(bm.getNRows() - i - 1)};
                WriteConsoleOutputCharacter(m_handle, terminalOutput.c_str(), terminalOutput.size(), coord, &charsWritten);
            }
        }

        void activate() const
        {
            SetConsoleActiveScreenBuffer(m_handle);
        }

    private:
        HANDLE m_handle;

    };

}


#endif // CUSTOSH_WINDOWSCONSOLESCREENBUFFER_H
