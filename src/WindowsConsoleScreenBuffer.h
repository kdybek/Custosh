#ifndef SNEK_WINDOWSCONSOLESCREENBUFFER_H
#define SNEK_WINDOWSCONSOLESCREENBUFFER_H


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
                for (unsigned int j = 0; j < bm.getNCols(); ++j) {
                    CHAR c = brightnessToASCII(bm(i, j));
                    DWORD charsWritten;
                    WriteConsoleOutputCharacter(m_handle, &c, 1,
                                                {static_cast<SHORT>(bm.getNRows() - i - 1), static_cast<SHORT>(j)},
                                                &charsWritten);
                }
            }
        }

        void activate() const
        {
            SetConsoleActiveScreenBuffer(m_handle);
        }

    private:
        HANDLE m_handle;

        static char brightnessToASCII(float brightness)
        {
            unsigned int idx = std::ceil(brightness * static_cast<float>(ASCIIByBrightness.size() - 1));
            return ASCIIByBrightness.at(idx);
        }
    };

}


#endif // SNEK_WINDOWSCONSOLESCREENBUFFER_H
