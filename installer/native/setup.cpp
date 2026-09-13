#include "common.hpp"
#include <commctrl.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <fstream>
#include <sstream>

namespace {
constexpr int Destination = 101, Browse = 102, Install = 103, Desktop = 104,
              StartMenu = 105, Status = 106, Progress = 107, OpenLog = 108;
HFONT font = nullptr, headingFont = nullptr;
HWND destinationEdit = nullptr, installButton = nullptr, browseButton = nullptr,
     statusLabel = nullptr, progressBar = nullptr, desktopBox = nullptr, menuBox = nullptr, logButton = nullptr;
std::filesystem::path installDirectory, sourceDirectory, logPath, resourceDirectory;
PROCESS_INFORMATION installProcess{};
bool installing = false, installed = false;
int scale = 96;

int px(int value) { return MulDiv(value, scale, 96); }

bool onlineSetup() {
    return FindResourceW(GetModuleHandleW(nullptr), MAKEINTRESOURCEW(103), RT_RCDATA) != nullptr;
}

void writeResource(int identifier, const std::filesystem::path& path) {
    const auto module = GetModuleHandleW(nullptr);
    const auto resource = FindResourceW(module, MAKEINTRESOURCEW(identifier), RT_RCDATA);
    const auto handle = resource ? LoadResource(module, resource) : nullptr;
    const auto bytes = handle ? LockResource(handle) : nullptr;
    const auto size = resource ? SizeofResource(module, resource) : 0;
    if (!bytes || !size) throw std::runtime_error("The setup download is incomplete. Download DarkFusionSetup.exe again.");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(static_cast<const char*>(bytes), size);
    if (!output) throw std::runtime_error("Setup could not prepare its temporary files.");
}

void prepareResources() {
    if (!resourceDirectory.empty()) return;
    GUID guid{};
    if (FAILED(CoCreateGuid(&guid))) throw std::runtime_error("Cannot create a setup session.");
    wchar_t suffix[40]{};
    StringFromGUID2(guid, suffix, 40);
    resourceDirectory = logPath.parent_path() / (std::wstring(L"DarkFusionSetup-") + suffix);
    if (!CreateDirectoryW(resourceDirectory.c_str(), nullptr)) throw std::runtime_error("Cannot create the temporary setup folder.");
    writeResource(101, resourceDirectory / L"install-standalone.ps1");
    writeResource(102, resourceDirectory / L"install-online.ps1");
    writeResource(103, resourceDirectory / L"download.json");
    writeResource(104, resourceDirectory / L"DarkFusion.exe");
}

void removeResources() {
    if (resourceDirectory.empty()) return;
    for (const auto* name : {L"install-standalone.ps1", L"install-online.ps1", L"download.json", L"DarkFusion.exe"})
        DeleteFileW((resourceDirectory / name).c_str());
    RemoveDirectoryW(resourceDirectory.c_str());
    resourceDirectory.clear();
}

void updateStageText() {
    std::ifstream input(logPath, std::ios::binary);
    std::string line, latest;
    while (std::getline(input, line)) {
        const auto marker = line.find("STAGE: ");
        if (marker != std::string::npos) latest = line.substr(marker + 7);
    }
    if (latest.empty()) return;
    const int length = MultiByteToWideChar(CP_UTF8, 0, latest.data(), static_cast<int>(latest.size()), nullptr, 0);
    if (!length) return;
    std::wstring text(length, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, latest.data(), static_cast<int>(latest.size()), text.data(), length);
    SetWindowTextW(statusLabel, text.c_str());
}

HWND add(HWND parent, const wchar_t* type, const wchar_t* title, DWORD style,
         int x, int y, int width, int height, int id = 0) {
    auto control = CreateWindowExW(type == std::wstring(L"EDIT") ? WS_EX_CLIENTEDGE : 0,
        type, title, WS_CHILD | WS_VISIBLE | style, px(x), px(y), px(width), px(height),
        parent, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), GetModuleHandleW(nullptr), nullptr);
    SendMessageW(control, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
    return control;
}

std::wstring textOf(HWND control) {
    std::wstring text(GetWindowTextLengthW(control) + 1, L'\0');
    const int length = GetWindowTextW(control, text.data(), static_cast<int>(text.size()));
    text.resize(length);
    return text;
}

std::filesystem::path defaultInstallDirectory() {
    PWSTR folder = nullptr;
    if (FAILED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &folder)))
        throw std::runtime_error("Cannot find your local application folder.");
    std::filesystem::path result(folder);
    CoTaskMemFree(folder);
    return result / L"Programs" / L"DarkFusion";
}

std::filesystem::path powershellPath() {
    std::vector<wchar_t> path(32768);
    const UINT length = GetSystemDirectoryW(path.data(), static_cast<UINT>(path.size()));
    if (!length || length >= path.size()) throw std::runtime_error("Cannot locate Windows PowerShell.");
    return std::filesystem::path(std::wstring(path.data(), length)) / L"WindowsPowerShell" / L"v1.0" / L"powershell.exe";
}

bool startInstallation(bool desktop, bool startMenu, std::wstring& error) {
    if (!onlineSetup()) for (const auto* file : {L"install-standalone.ps1", L"payload.zip", L"payload.json"}) {
        if (!std::filesystem::is_regular_file(sourceDirectory / file)) {
            error = std::wstring(L"The download is incomplete: ") + file +
                L" is missing. Extract all of the installer files into the same folder.";
            return false;
        }
    }
    if (installDirectory.empty() || !installDirectory.is_absolute()) {
        error = L"Choose a full folder path, such as D:\\Applications\\DarkFusion.";
        return false;
    }
    std::vector<std::wstring> arguments{L"-NoLogo", L"-NoProfile", L"-NonInteractive", L"-ExecutionPolicy", L"Bypass", L"-File"};
    auto workingDirectory = sourceDirectory;
    if (onlineSetup()) {
        try { prepareResources(); }
        catch (const std::exception&) {
            error = L"Setup could not prepare its temporary files. Check that your temporary folder is writable, then download and run setup again.";
            return false;
        }
        workingDirectory = resourceDirectory;
        arguments.push_back((resourceDirectory / L"install-online.ps1").wstring());
    } else {
        arguments.insert(arguments.end(), {(sourceDirectory / L"install-standalone.ps1").wstring(),
            L"-PackagePath", (sourceDirectory / L"payload.zip").wstring(),
            L"-ManifestPath", (sourceDirectory / L"payload.json").wstring()});
    }
    arguments.insert(arguments.end(), {L"-InstallDirectory", installDirectory.wstring(), L"-LogPath", logPath.wstring()});
    if (desktop) arguments.push_back(L"-DesktopShortcut");
    if (startMenu) arguments.push_back(L"-StartMenuShortcut");
    if (!df::startProcess(powershellPath(), arguments, workingDirectory, installProcess)) {
        error = L"Windows could not start the installer.\n\n" + df::errorMessage();
        return false;
    }
    return true;
}

void browse(HWND owner) {
    IFileOpenDialog* dialog = nullptr;
    if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog)))) return;
    DWORD options = 0;
    dialog->GetOptions(&options);
    dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_NOCHANGEDIR);
    dialog->SetTitle(L"Choose a writable folder for DarkFusion");
    dialog->SetOkButtonLabel(L"Use this folder");
    IShellItem* initial = nullptr;
    const auto current = textOf(destinationEdit);
    if (SUCCEEDED(SHCreateItemFromParsingName(current.c_str(), nullptr, IID_PPV_ARGS(&initial)))) {
        dialog->SetFolder(initial);
        initial->Release();
    }
    if (SUCCEEDED(dialog->Show(owner))) {
        IShellItem* selected = nullptr;
        if (SUCCEEDED(dialog->GetResult(&selected))) {
            PWSTR path = nullptr;
            if (SUCCEEDED(selected->GetDisplayName(SIGDN_FILESYSPATH, &path))) {
                SetWindowTextW(destinationEdit, path);
                CoTaskMemFree(path);
            }
            selected->Release();
        }
    }
    dialog->Release();
}

void begin(HWND owner) {
    if (installed) {
        PROCESS_INFORMATION process{};
        if (!df::startProcess(installDirectory / L"DarkFusion.exe", {}, installDirectory, process)) {
            const auto error = L"Could not open DarkFusion.\n\n" + df::errorMessage();
            MessageBoxW(owner, error.c_str(), L"DarkFusion", MB_OK | MB_ICONERROR);
        } else {
            df::closeProcess(process);
            DestroyWindow(owner);
        }
        return;
    }
    installDirectory = textOf(destinationEdit);
    std::wstring error;
    if (!startInstallation(SendMessageW(desktopBox, BM_GETCHECK, 0, 0) == BST_CHECKED,
                           SendMessageW(menuBox, BM_GETCHECK, 0, 0) == BST_CHECKED, error)) {
        MessageBoxW(owner, error.c_str(), L"DarkFusion setup", MB_OK | MB_ICONERROR);
        return;
    }
    installing = true;
    for (auto control : {destinationEdit, browseButton, installButton, desktopBox, menuBox}) EnableWindow(control, FALSE);
    SetWindowTextW(statusLabel, L"Preparing your installation...");
    SendMessageW(progressBar, PBM_SETMARQUEE, TRUE, 40);
    SetTimer(owner, 1, 250, nullptr);
}

LRESULT CALLBACK windowProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_CREATE: {
        scale = static_cast<int>(GetDpiForWindow(window));
        font = CreateFontW(-px(15), 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
            OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
        headingFont = CreateFontW(-px(25), 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
            OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
        auto heading = add(window, L"STATIC", L"Install DarkFusion", 0, 24, 20, 565, 35);
        SendMessageW(heading, WM_SETFONT, reinterpret_cast<WPARAM>(headingFont), TRUE);
        add(window, L"STATIC", onlineSetup() ? L"Includes the app, Python runtime, SAM3 and GroundingDINO models. Setup downloads and installs them automatically." : L"Includes its own Python runtime. Your other Python and Conda environments stay separate.", 0, 24, 65, 555, 48);
        add(window, L"STATIC", L"Install to a writable folder", 0, 24, 125, 555, 22);
        destinationEdit = add(window, L"EDIT", installDirectory.c_str(), WS_TABSTOP | ES_AUTOHSCROLL, 24, 153, 448, 31, Destination);
        browseButton = add(window, L"BUTTON", L"Browse...", WS_TABSTOP | BS_PUSHBUTTON, 482, 152, 105, 33, Browse);
        menuBox = add(window, L"BUTTON", L"Add a Start menu shortcut", WS_TABSTOP | BS_AUTOCHECKBOX, 24, 203, 290, 25, StartMenu);
        SendMessageW(menuBox, BM_SETCHECK, BST_CHECKED, 0);
        desktopBox = add(window, L"BUTTON", L"Add a desktop shortcut", WS_TABSTOP | BS_AUTOCHECKBOX, 321, 203, 267, 25, Desktop);
        statusLabel = add(window, L"STATIC", onlineSetup() ? L"Download: about 10.2 GB. Allow 35 GB free during setup.\nChoose a folder, then click Install." : L"Ready to install. Keep the installer files together until setup finishes.", 0, 24, 247, 560, 48, Status);
        progressBar = add(window, PROGRESS_CLASSW, L"", PBS_MARQUEE, 24, 309, 563, 15, Progress);
        logButton = add(window, L"BUTTON", L"View install log", WS_TABSTOP | BS_PUSHBUTTON, 24, 346, 150, 35, OpenLog);
        installButton = add(window, L"BUTTON", L"Install", WS_TABSTOP | BS_DEFPUSHBUTTON, 437, 346, 150, 35, Install);
        return 0;
    }
    case WM_COMMAND:
        switch (LOWORD(wParam)) {
        case Browse: browse(window); break;
        case Install: begin(window); break;
        case OpenLog:
            if (std::filesystem::is_regular_file(logPath)) ShellExecuteW(window, L"open", logPath.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
            else MessageBoxW(window, L"The log will be available once installation starts.", L"DarkFusion setup", MB_OK | MB_ICONINFORMATION);
            break;
        }
        return 0;
    case WM_TIMER:
        if (installing) updateStageText();
        if (installing && WaitForSingleObject(installProcess.hProcess, 0) == WAIT_OBJECT_0) {
            DWORD code = 1;
            GetExitCodeProcess(installProcess.hProcess, &code);
            df::closeProcess(installProcess);
            KillTimer(window, 1);
            installing = false;
            SendMessageW(progressBar, PBM_SETMARQUEE, FALSE, 0);
            EnableWindow(installButton, TRUE);
            if (code == 0) {
                installed = true;
                SetWindowTextW(statusLabel, L"DarkFusion is installed. Open it now or use your shortcut whenever you're ready.");
                SetWindowTextW(installButton, L"Open DarkFusion");
            } else {
                SetWindowTextW(statusLabel, L"Installation did not finish. View the install log for details, then retry.");
                for (auto control : {destinationEdit, browseButton, desktopBox, menuBox}) EnableWindow(control, TRUE);
                const auto error = L"Installation exited with code " + std::to_wstring(code) + L".\n\nLog: " + logPath.wstring();
                MessageBoxW(window, error.c_str(), L"DarkFusion setup", MB_OK | MB_ICONERROR);
            }
        }
        return 0;
    case WM_CLOSE:
        if (installing) MessageBoxW(window, L"Please wait for installation to finish before closing setup.", L"DarkFusion setup", MB_OK | MB_ICONINFORMATION);
        else DestroyWindow(window);
        return 0;
    case WM_DESTROY:
        removeResources();
        if (font) DeleteObject(font);
        if (headingFont) DeleteObject(headingFont);
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(window, message, wParam, lParam);
}
} // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int show) {
    bool quiet = false;
    try {
        const auto arguments = df::arguments();
        for (const auto& argument : arguments) if (argument == L"--quiet") quiet = true;
        installDirectory = defaultInstallDirectory();
        sourceDirectory = df::executableDirectory();
        for (size_t i = 0; i < arguments.size(); ++i) {
            if (arguments[i] == L"--quiet") continue;
            if (arguments[i] == L"--install-dir" && i + 1 < arguments.size()) installDirectory = arguments[++i];
            else {
                if (!quiet) MessageBoxW(nullptr, L"Usage: DarkFusionSetup.exe [--install-dir PATH] [--quiet]", L"DarkFusion setup", MB_OK | MB_ICONINFORMATION);
                return 2;
            }
        }
        wchar_t temp[MAX_PATH + 1]{};
        if (!GetTempPathW(MAX_PATH, temp)) throw std::runtime_error("Cannot locate the temporary folder.");
        logPath = std::filesystem::path(temp) / (L"DarkFusion-install-" + std::to_wstring(GetCurrentProcessId()) + L".log");
        if (quiet) {
            std::wstring error;
            if (!startInstallation(false, true, error)) {
                HANDLE log = CreateFileW(logPath.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
                if (log != INVALID_HANDLE_VALUE) {
                    DWORD written = 0;
                    const wchar_t bom = 0xFEFF;
                    WriteFile(log, &bom, sizeof(bom), &written, nullptr);
                    WriteFile(log, error.data(), static_cast<DWORD>(error.size() * sizeof(wchar_t)), &written, nullptr);
                    CloseHandle(log);
                }
                return 3;
            }
            WaitForSingleObject(installProcess.hProcess, INFINITE);
            DWORD code = 1;
            GetExitCodeProcess(installProcess.hProcess, &code);
            df::closeProcess(installProcess);
            removeResources();
            return static_cast<int>(code);
        }
        CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
        INITCOMMONCONTROLSEX controls{sizeof(controls), ICC_PROGRESS_CLASS | ICC_STANDARD_CLASSES};
        InitCommonControlsEx(&controls);
        WNDCLASSW windowClass{};
        windowClass.lpfnWndProc = windowProc;
        windowClass.hInstance = instance;
        windowClass.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        windowClass.hIcon = LoadIconW(instance, MAKEINTRESOURCEW(201));
        windowClass.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
        windowClass.lpszClassName = L"DarkFusionSetupWindow";
        RegisterClassW(&windowClass);
        scale = static_cast<int>(GetDpiForSystem());
        RECT bounds{0, 0, px(612), px(406)};
        constexpr DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
        AdjustWindowRectExForDpi(&bounds, style, FALSE, 0, scale);
        auto window = CreateWindowExW(0, windowClass.lpszClassName, L"DarkFusion Setup", style,
            CW_USEDEFAULT, CW_USEDEFAULT, bounds.right - bounds.left, bounds.bottom - bounds.top,
            nullptr, nullptr, instance, nullptr);
        if (!window) throw std::runtime_error("Cannot create the installer window.");
        const UINT dpi = GetDpiForWindow(window);
        const auto smallIcon = static_cast<HICON>(LoadImageW(instance, MAKEINTRESOURCEW(201), IMAGE_ICON,
            GetSystemMetricsForDpi(SM_CXSMICON, dpi), GetSystemMetricsForDpi(SM_CYSMICON, dpi), LR_SHARED));
        const auto largeIcon = static_cast<HICON>(LoadImageW(instance, MAKEINTRESOURCEW(201), IMAGE_ICON,
            GetSystemMetricsForDpi(SM_CXICON, dpi), GetSystemMetricsForDpi(SM_CYICON, dpi), LR_SHARED));
        SendMessageW(window, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(smallIcon));
        SendMessageW(window, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(largeIcon));
        ShowWindow(window, show);
        MSG message{};
        while (GetMessageW(&message, nullptr, 0, 0) > 0) {
            if (!IsDialogMessageW(window, &message)) { TranslateMessage(&message); DispatchMessageW(&message); }
        }
        CoUninitialize();
        return static_cast<int>(message.wParam);
    } catch (const std::exception& error) {
        if (!quiet) MessageBoxA(nullptr, error.what(), "DarkFusion setup", MB_OK | MB_ICONERROR);
        return 1;
    }
}
