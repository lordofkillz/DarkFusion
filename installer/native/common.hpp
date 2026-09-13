#pragma once
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace df {
inline std::wstring errorMessage(DWORD error = GetLastError()) {
    wchar_t* text = nullptr;
    FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                       FORMAT_MESSAGE_IGNORE_INSERTS,
                   nullptr, error, 0, reinterpret_cast<wchar_t*>(&text), 0, nullptr);
    std::wstring result = text ? text : L"Windows error";
    if (text) LocalFree(text);
    return result + L" (" + std::to_wstring(error) + L")";
}

inline std::filesystem::path executableDirectory() {
    std::vector<wchar_t> buffer(512);
    for (;;) {
        const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (!length) throw std::runtime_error("Cannot locate the executable.");
        if (length < buffer.size()) return std::filesystem::path(std::wstring(buffer.data(), length)).parent_path();
        buffer.resize(buffer.size() * 2);
    }
}

// CommandLineToArgvW/MSVC argument quoting. Never send application paths through a shell.
inline std::wstring quoteArgument(const std::wstring& value) {
    std::wstring result = L"\"";
    size_t backslashes = 0;
    for (const wchar_t c : value) {
        if (c == L'\\') { ++backslashes; continue; }
        if (c == L'\"') {
            result.append(backslashes * 2 + 1, L'\\');
            result += L'\"';
        } else {
            result.append(backslashes, L'\\');
            result += c;
        }
        backslashes = 0;
    }
    result.append(backslashes * 2, L'\\');
    return result + L'\"';
}

inline std::wstring commandLine(const std::filesystem::path& executable,
                                const std::vector<std::wstring>& arguments) {
    std::wstring result = quoteArgument(executable.wstring());
    for (const auto& argument : arguments) result += L" " + quoteArgument(argument);
    return result;
}

inline std::vector<std::wstring> arguments() {
    int count = 0;
    auto values = CommandLineToArgvW(GetCommandLineW(), &count);
    if (!values) throw std::runtime_error("Cannot read command line.");
    std::vector<std::wstring> result(values + 1, values + count);
    LocalFree(values);
    return result;
}

inline bool startProcess(const std::filesystem::path& executable,
                         const std::vector<std::wstring>& arguments,
                         const std::filesystem::path& directory,
                         PROCESS_INFORMATION& process,
                         DWORD flags = CREATE_NO_WINDOW,
                         wchar_t* environment = nullptr) {
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    auto command = commandLine(executable, arguments);
    return CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, FALSE,
                          flags | (environment ? CREATE_UNICODE_ENVIRONMENT : 0),
                          environment, directory.c_str(), &startup, &process) != FALSE;
}

inline void closeProcess(PROCESS_INFORMATION& process) {
    if (process.hThread) CloseHandle(process.hThread);
    if (process.hProcess) CloseHandle(process.hProcess);
    process = {};
}

inline std::wstring environmentVariable(const wchar_t* name) {
    DWORD size = GetEnvironmentVariableW(name, nullptr, 0);
    if (!size) return {};
    std::wstring value(size, L'\0');
    DWORD length = GetEnvironmentVariableW(name, value.data(), size);
    value.resize(length);
    return value;
}

struct CaseInsensitiveLess {
    bool operator()(const std::wstring& left, const std::wstring& right) const {
        return _wcsicmp(left.c_str(), right.c_str()) < 0;
    }
};

inline std::vector<wchar_t> privateRuntimeEnvironment(const std::filesystem::path& runtime) {
    std::map<std::wstring, std::wstring, CaseInsensitiveLess> values;
    auto inherited = GetEnvironmentStringsW();
    if (!inherited) throw std::runtime_error("Cannot read the environment.");
    for (const wchar_t* item = inherited; *item; item += wcslen(item) + 1) {
        std::wstring entry(item);
        const auto separator = entry.find(L'=', entry[0] == L'=' ? 1 : 0);
        if (separator != std::wstring::npos) values[entry.substr(0, separator)] = entry.substr(separator + 1);
    }
    FreeEnvironmentStringsW(inherited);
    for (const auto* name : {L"PYTHONHOME", L"PYTHONPATH", L"VIRTUAL_ENV", L"CONDA_DEFAULT_ENV",
                             L"QT_PLUGIN_PATH", L"QT_QPA_PLATFORM_PLUGIN_PATH", L"QML2_IMPORT_PATH"}) values.erase(name);
    values[L"PYTHONNOUSERSITE"] = L"1";
    values[L"CONDA_PREFIX"] = runtime.wstring();
    // PyQt5's embedded qt.conf can lose Unicode characters in the runtime path.
    // Supply the private plugin directories through the Unicode environment.
    const auto plugins = runtime / L"Lib" / L"site-packages" / L"PyQt5" / L"Qt5" / L"plugins";
    values[L"QT_PLUGIN_PATH"] = plugins.wstring();
    values[L"QT_QPA_PLATFORM_PLUGIN_PATH"] = (plugins / L"platforms").wstring();
    values[L"PATH"] = runtime.wstring() + L";" + (runtime / L"Library" / L"mingw-w64" / L"bin").wstring() +
                       L";" + (runtime / L"Library" / L"usr" / L"bin").wstring() +
                       L";" + (runtime / L"Library" / L"bin").wstring() +
                       L";" + (runtime / L"Scripts").wstring() + L";" + (runtime / L"bin").wstring() + L";" + values[L"PATH"];
    std::vector<wchar_t> block;
    for (const auto& [name, value] : values) {
        const auto entry = name + L"=" + value;
        block.insert(block.end(), entry.begin(), entry.end());
        block.push_back(L'\0');
    }
    block.push_back(L'\0');
    return block;
}
} // namespace df
