#include "common.hpp"
#include <iostream>

namespace {
std::vector<std::wstring> trickyArguments() {
    return {L"", L"ordinary", L"space here", L"D:\\trailing folder\\", L"a\"b", L"two\\\\\"quotes",
            L"ampersand & semicolon ; dollar $ backtick ` apostrophe '", L"Unicode \u03A9 \u4E2D\u6587"};
}
}

int wmain() {
    try {
        const auto arguments = df::arguments();
        if (!arguments.empty() && arguments[0] == L"--child") {
            const std::vector<std::wstring> received(arguments.begin() + 1, arguments.end());
            if (received != trickyArguments()) return 11;
            if (df::environmentVariable(L"PYTHONNOUSERSITE") != L"1") return 12;
            for (auto name : {L"PYTHONHOME", L"PYTHONPATH", L"VIRTUAL_ENV"})
                if (!df::environmentVariable(name).empty()) return 13;
            if (df::environmentVariable(L"CONDA_PREFIX") != L"D:\\private runtime \u03A9 & things") return 14;
            if (df::environmentVariable(L"PATH").find(L"D:\\private runtime \u03A9 & things;") != 0) return 15;
            const auto plugins = std::filesystem::path(L"D:\\private runtime \u03A9 & things") / L"Lib" / L"site-packages" / L"PyQt5" / L"Qt5" / L"plugins";
            if (df::environmentVariable(L"QT_PLUGIN_PATH") != plugins.wstring()) return 16;
            if (df::environmentVariable(L"QT_QPA_PLATFORM_PLUGIN_PATH") != (plugins / L"platforms").wstring()) return 17;
            return 37;
        }
        auto cases = trickyArguments();
        auto command = df::commandLine(L"C:\\program folder\\app.exe", cases);
        int count = 0;
        auto parsed = CommandLineToArgvW(command.c_str(), &count);
        if (!parsed || count != static_cast<int>(cases.size() + 1)) throw std::runtime_error("Quote roundtrip count failed.");
        for (size_t i = 0; i < cases.size(); ++i) if (parsed[i + 1] != cases[i]) throw std::runtime_error("Quote roundtrip value failed.");
        LocalFree(parsed);
        for (auto name : {L"PYTHONHOME", L"PYTHONPATH", L"QT_PLUGIN_PATH", L"QT_QPA_PLATFORM_PLUGIN_PATH", L"VIRTUAL_ENV"})
            SetEnvironmentVariableW(name, L"foreign environment poison");
        SetEnvironmentVariableW(L"CONDA_PREFIX", L"C:\\foreign conda");
        auto environment = df::privateRuntimeEnvironment(L"D:\\private runtime \u03A9 & things");
        cases.insert(cases.begin(), L"--child");
        PROCESS_INFORMATION process{};
        const auto directory = df::executableDirectory();
        if (!df::startProcess(directory / L"native smoke.exe", cases, directory, process, CREATE_NO_WINDOW, environment.data()))
            throw std::runtime_error("CreateProcess smoke failed.");
        if (WaitForSingleObject(process.hProcess, 10000) != WAIT_OBJECT_0) {
            TerminateProcess(process.hProcess, 99);
            df::closeProcess(process);
            throw std::runtime_error("Child did not finish.");
        }
        DWORD code = 0;
        GetExitCodeProcess(process.hProcess, &code);
        df::closeProcess(process);
        if (code != 37) throw std::runtime_error("Real child argv/environment/exit-code verification failed.");
        PROCESS_INFORMATION missing{};
        if (df::startProcess(directory / L"does-not-exist.exe", {}, directory, missing)) throw std::runtime_error("Missing executable should fail.");
        std::cout << "PASS: Windows argv quoting, real Unicode child process, private runtime environment, exit status, missing executable.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
