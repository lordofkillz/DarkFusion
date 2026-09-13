#include "common.hpp"

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    bool console = false;
    bool verify = false;
    try {
        for (const auto& argument : df::arguments()) {
            if (argument == L"--console") console = true;
            else if (argument == L"--verify") verify = true;
            else {
                MessageBoxW(nullptr, L"Usage: DarkFusion.exe [--console] [--verify]", L"DarkFusion", MB_OK | MB_ICONINFORMATION);
                return 2;
            }
        }
        const auto root = df::executableDirectory();
        const auto runtime = root / L"runtime";
        // Training workers use sys.executable and piped stdout; pythonw.exe would lose that output.
        const auto python = runtime / L"python.exe";
        const auto script = verify ? root / L"app" / L"scripts" / L"verify_install.py"
                                   : root / L"app" / L"UltraDarkFusion" / L"UltraDarkFusion_v5.2.py";
        const auto workingDirectory = root / L"app" / L"UltraDarkFusion";
        if (!std::filesystem::is_regular_file(python) || !std::filesystem::is_regular_file(script) ||
            !std::filesystem::is_directory(workingDirectory)) {
            if (!verify) MessageBoxW(nullptr,
                L"DarkFusion's private runtime or application files are missing.\n\nRun DarkFusionSetup.exe to install into a new, empty folder.",
                L"DarkFusion could not start", MB_OK | MB_ICONERROR);
            return 3;
        }
        auto environment = df::privateRuntimeEnvironment(runtime);
        PROCESS_INFORMATION process{};
        std::vector<std::wstring> arguments{L"-s", script.wstring()};
        if (!df::startProcess(python, arguments, workingDirectory, process,
                              console ? CREATE_NEW_CONSOLE : CREATE_NO_WINDOW, environment.data())) {
            const auto error = L"Could not start DarkFusion.\n\n" + df::errorMessage();
            if (!verify) MessageBoxW(nullptr, error.c_str(), L"DarkFusion could not start", MB_OK | MB_ICONERROR);
            return 4;
        }
        if (console || verify) {
            WaitForSingleObject(process.hProcess, INFINITE);
            DWORD code = 1;
            GetExitCodeProcess(process.hProcess, &code);
            df::closeProcess(process);
            return static_cast<int>(code);
        }
        df::closeProcess(process);
        return 0;
    } catch (const std::exception& error) {
        if (!verify) MessageBoxA(nullptr, error.what(), "DarkFusion could not start", MB_OK | MB_ICONERROR);
        return 1;
    }
}
