#ifndef CUSTOSH_WINDOWS_CONSOLE_SCREEN_BUFFER_H
#define CUSTOSH_WINDOWS_CONSOLE_SCREEN_BUFFER_H


#define NOMINMAX

#include <Windows.h>
#include <iostream>

#include "../custosh_except.h"
#include "../utils.h"

namespace Custosh
{
    class WindowsConsoleScreenBuffer
    {
    public:
        WindowsConsoleScreenBuffer() : m_handle(CreateConsoleScreenBuffer(
                GENERIC_WRITE | GENERIC_READ,   // Access rights
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

        WindowsConsoleScreenBuffer(WindowsConsoleScreenBuffer&& other) noexcept
                : m_handle(std::exchange(other.m_handle, INVALID_HANDLE_VALUE))
        {
        }

        WindowsConsoleScreenBuffer& operator=(WindowsConsoleScreenBuffer&& other) noexcept
        {
            if (this != &other) {
                CloseHandle(m_handle);
                m_handle = std::exchange(other.m_handle, INVALID_HANDLE_VALUE);
            }
            return *this;
        }

        void draw(const char* str, unsigned int rows, unsigned int cols)
        {
            for (unsigned int i = 0; i < rows; ++i) {
                DWORD charsWritten;
                COORD coord = {0, static_cast<SHORT>(rows - i - 1)};
                WriteConsoleOutputCharacter(m_handle, str + i * cols, cols, coord, &charsWritten);
                // Number of written chars is ignored to not waste time.
            }
        }

        void activate() const
        {
            SetConsoleActiveScreenBuffer(m_handle);
        }

        [[nodiscard]] Vector2<unsigned int> getWindowDimensions()
        {
            CONSOLE_SCREEN_BUFFER_INFO csbi;
            if (!GetConsoleScreenBufferInfo(m_handle, &csbi)) {
                throw CustoshException("error getting console screen buffer info");
            }

            unsigned int rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
            unsigned int cols = csbi.srWindow.Right - csbi.srWindow.Left + 1;

            return {cols, rows};
        }

    private:
        HANDLE m_handle;

    }; // WindowsConsoleScreenBuffer

}


#endif // CUSTOSH_WINDOWS_CONSOLE_SCREEN_BUFFER_H
