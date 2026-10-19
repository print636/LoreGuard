[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Setup', 'Start', 'Stop', 'Status', 'CodexLogin', 'SiwcBind', 'Models', 'StartLoreGuard', 'StopLoreGuard')]
    [string]$Action,
    [string]$CredentialFileName
)

# Credentials and binaries never belong in the repository or the shared test app.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$taskVersion = '8.0.33'
$taskArchiveHash = '6a1e04025710f01c528b88c4ae26ea1a332aabc27c7387ae3527107b9096f8a5'
$taskChecksumHash = 'a69d87de882f6b8bb84117558dda23c99515ee1f9f38e82095e7e5ae5239bcfc'
$taskOwner = 'LoreGuardLocalProGateway/v1'
$taskDesktop = [Environment]::GetFolderPath('Desktop')
$taskRuntime = [IO.Path]::GetFullPath((Join-Path $taskDesktop 'dev\tools\LoreGuard-Pro-Gateway'))
$taskAuthDir = Join-Path $taskRuntime 'auth'
$taskLogDir = Join-Path $taskRuntime 'logs'
$taskConfig = Join-Path $taskRuntime 'config.yaml'
$taskKeyPath = Join-Path $taskRuntime 'client-api-key.txt'
$taskManifestPath = Join-Path $taskRuntime 'installation.json'
$taskStatePath = Join-Path $taskRuntime 'process.json'
$taskOAuthStatePath = Join-Path $taskRuntime 'oauth-process.json'
$taskAppStatePath = Join-Path $taskRuntime 'loreguard-process.json'
$taskRepository = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$taskAppPython = Join-Path $taskRepository '.venv\Scripts\python.exe'
$taskAppData = Join-Path $taskRuntime 'loreguard-data'
$taskExe = Join-Path $taskRuntime "bin\$taskVersion\cli-proxy-api.exe"
$taskBaseUrl = 'http://127.0.0.1:8317/v1'

function Assert-NoReparsePath([string]$Path) {
    $taskWalk = [IO.Path]::GetFullPath($Path)
    while ($taskWalk) {
        if (Test-Path -LiteralPath $taskWalk) {
            if ((Get-Item -LiteralPath $taskWalk -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw 'Runtime path contains a junction/symlink; choose a normal local Desktop directory.'
            }
        }
        $taskParent = [IO.Directory]::GetParent($taskWalk)
        if ($null -eq $taskParent) { break }
        $taskWalk = $taskParent.FullName
    }
}

function Set-PrivateDirectoryAcl([string]$Path) {
    $taskCurrentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User
    $taskAllowedSids = @($taskCurrentSid.Value, 'S-1-5-18', 'S-1-5-32-544')
    $taskExistingAcl = Get-Acl -LiteralPath $Path
    $taskExistingPrivate = $taskExistingAcl.AreAccessRulesProtected
    foreach ($taskExistingRule in $taskExistingAcl.Access) {
        $taskExistingSid = $taskExistingRule.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value
        if ($taskExistingRule.AccessControlType -eq 'Allow' -and $taskExistingSid -notin $taskAllowedSids) { $taskExistingPrivate = $false }
    }
    # Do not rewrite a correct DACL: some Windows tokens cannot reapply owner /
    # audit sections even though the required private access rules already hold.
    if ($taskExistingPrivate) { return }
    $taskAcl = New-Object Security.AccessControl.DirectorySecurity
    $taskAcl.SetAccessRuleProtection($true, $false)
    $taskAcl.SetOwner($taskCurrentSid)
    foreach ($taskSid in @($taskCurrentSid, (New-Object Security.Principal.SecurityIdentifier('S-1-5-18')), (New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')))) {
        $taskRule = New-Object Security.AccessControl.FileSystemAccessRule($taskSid, 'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow')
        $taskAcl.AddAccessRule($taskRule)
    }
    Set-Acl -LiteralPath $Path -AclObject $taskAcl
}

function Write-PrivateText([string]$Path, [string]$Value) {
    # Used only for generated runtime state, never for repository source edits.
    [IO.File]::WriteAllText($Path, $Value, (New-Object Text.UTF8Encoding($false)))
}

function Assert-Installation {
    Assert-NoReparsePath $taskRuntime
    if (!(Test-Path -LiteralPath $taskManifestPath -PathType Leaf)) { throw 'Run -Action Setup first.' }
    $taskManifest = Get-Content -LiteralPath $taskManifestPath -Raw | ConvertFrom-Json
    if ($taskManifest.owner -ne $taskOwner -or $taskManifest.version -ne $taskVersion -or $taskManifest.runtime -ne $taskRuntime) {
        throw 'Runtime ownership/version mismatch; do not reuse an unrelated directory.'
    }
    if (!(Test-Path -LiteralPath $taskExe -PathType Leaf) -or (Get-FileHash -LiteralPath $taskExe -Algorithm SHA256).Hash -ne $taskManifest.exe_sha256) {
        throw 'Pinned gateway executable is missing or changed; refusing to execute it.'
    }
    foreach ($taskPath in @($taskRuntime, $taskAuthDir, $taskLogDir)) {
        Assert-NoReparsePath $taskPath
        $taskAcl = Get-Acl -LiteralPath $taskPath
        $taskAllowedSids = @([Security.Principal.WindowsIdentity]::GetCurrent().User.Value, 'S-1-5-18', 'S-1-5-32-544')
        if (!$taskAcl.AreAccessRulesProtected) { throw 'Runtime ACL inheritance must be disabled.' }
        foreach ($taskRule in $taskAcl.Access) {
            $taskSid = $taskRule.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value
            if ($taskRule.AccessControlType -eq 'Allow' -and $taskSid -notin $taskAllowedSids) { throw 'Runtime ACL grants another identity access; refusing.' }
        }
    }
    foreach ($taskPath in @($taskConfig, $taskKeyPath)) {
        if (!(Test-Path -LiteralPath $taskPath -PathType Leaf)) { throw 'Generated private configuration is missing.' }
        Assert-NoReparsePath $taskPath
    }
    # Refuse edited configs that broaden the listener, management, auth root, or client key.
    if ((Get-FileHash -LiteralPath $taskConfig -Algorithm SHA256).Hash -ne $taskManifest.config_sha256 -or
        (Get-FileHash -LiteralPath $taskKeyPath -Algorithm SHA256).Hash -ne $taskManifest.key_sha256) {
        throw 'Private configuration changed; the local-only launcher refuses unreviewed edits.'
    }
}

function Get-OwnedProcess([string]$StatePath) {
    if (!(Test-Path -LiteralPath $StatePath -PathType Leaf)) { return $null }
    Assert-NoReparsePath $StatePath
    $taskState = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
    $taskExpectedExecutable = $taskExe
    if ($StatePath -eq $taskAppStatePath) { $taskExpectedExecutable = $taskAppPython }
    if ($taskState.owner -ne $taskOwner -or $taskState.runtime -ne $taskRuntime -or $taskState.executable -ne $taskExpectedExecutable) {
        throw 'Process state does not belong to this gateway.'
    }
    $taskProcess = Get-Process -Id $taskState.pid -ErrorAction SilentlyContinue
    if ($null -eq $taskProcess) { return $null }
    $taskCim = Get-CimInstance Win32_Process -Filter "ProcessId=$($taskState.pid)"
    if ($taskProcess.StartTime.ToUniversalTime().Ticks -ne [long]$taskState.started_ticks -or
        $taskCim.ExecutablePath -ne $taskExpectedExecutable) {
        throw 'PID was reused or command differs; refusing to control that process.'
    }
    if ($StatePath -eq $taskAppStatePath) {
        if ($taskState.repository -ne $taskRepository -or !$taskCim.CommandLine.Contains('scripts.local_pro_gateway_app:create_app') -or
            !$taskCim.CommandLine.Contains($taskRepository) -or !$taskCim.CommandLine.Contains('--port 8787')) {
            throw 'Application process command/ownership differs; refusing to control it.'
        }
    } elseif (!$taskCim.CommandLine.Contains($taskConfig)) { throw 'Gateway process config differs; refusing to control it.' }
    return $taskProcess
}

function Stop-OwnedProcess([string]$StatePath) {
    $taskProcess = Get-OwnedProcess $StatePath
    if ($null -ne $taskProcess) {
        Stop-Process -Id $taskProcess.Id -ErrorAction Stop
        $taskProcess.WaitForExit(5000) | Out-Null
        if (!$taskProcess.HasExited) { throw 'Owned gateway process did not stop.' }
    }
    if (Test-Path -LiteralPath $StatePath) { Remove-Item -LiteralPath $StatePath }
}

function Get-AppBaseExecutable {
    # The Windows venv executable may be a launcher with a separate real worker.
    $taskBaseExecutable = (& $taskAppPython -I -c 'import sys; print(sys._base_executable)' | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or ![IO.Path]::IsPathRooted($taskBaseExecutable) -or !(Test-Path -LiteralPath $taskBaseExecutable -PathType Leaf)) {
        throw 'Cannot identify the repository virtual environment base interpreter.'
    }
    return [IO.Path]::GetFullPath($taskBaseExecutable)
}

function Get-OwnedAppWorker {
    if (!(Test-Path -LiteralPath $taskAppStatePath -PathType Leaf)) { return $null }
    $taskParentProcess = Get-OwnedProcess $taskAppStatePath
    $taskState = Get-Content -LiteralPath $taskAppStatePath -Raw | ConvertFrom-Json
    $taskBaseExecutable = Get-AppBaseExecutable
    if ($taskState.PSObject.Properties['worker_pid']) {
        $taskWorkerProcess = Get-Process -Id $taskState.worker_pid -ErrorAction SilentlyContinue
        if ($null -eq $taskWorkerProcess) { return $null }
        $taskWorkerCim = Get-CimInstance Win32_Process -Filter "ProcessId=$($taskState.worker_pid)"
        if ($taskWorkerProcess.StartTime.ToUniversalTime().Ticks -ne [long]$taskState.worker_started_ticks -or
            $taskState.worker_executable -ne $taskBaseExecutable) { throw 'Recorded application worker PID was reused or interpreter differs.' }
    } else {
        # Compatibility with an owned instance started before worker discovery.
        if ($null -eq $taskParentProcess) { return $null }
        $taskCandidates = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($taskState.pid)" | Where-Object {
            $_.ExecutablePath -eq $taskBaseExecutable -and $_.CommandLine.Contains('scripts.local_pro_gateway_app:create_app') -and
            $_.CommandLine.Contains($taskRepository) -and $_.CommandLine.Contains('--port 8787')
        })
        if ($taskCandidates.Count -eq 0) { return $null }
        if ($taskCandidates.Count -ne 1) { throw 'Ambiguous application workers; no process will be stopped.' }
        $taskWorkerCim = $taskCandidates[0]
        $taskWorkerProcess = Get-Process -Id $taskWorkerCim.ProcessId -ErrorAction Stop
        $taskState | Add-Member -NotePropertyName worker_pid -NotePropertyValue $taskWorkerProcess.Id
        $taskState | Add-Member -NotePropertyName worker_started_ticks -NotePropertyValue $taskWorkerProcess.StartTime.ToUniversalTime().Ticks
        $taskState | Add-Member -NotePropertyName worker_executable -NotePropertyValue $taskBaseExecutable
        Write-PrivateText $taskAppStatePath ($taskState | ConvertTo-Json)
    }
    if ($taskWorkerCim.ParentProcessId -ne $taskState.pid -or $taskWorkerCim.ExecutablePath -ne $taskBaseExecutable -or
        !$taskWorkerCim.CommandLine.Contains('scripts.local_pro_gateway_app:create_app') -or
        !$taskWorkerCim.CommandLine.Contains($taskRepository) -or !$taskWorkerCim.CommandLine.Contains('--port 8787')) {
        throw 'Application worker ancestry/command differs; refusing to control it.'
    }
    return $taskWorkerProcess
}

function Stop-OwnedApp {
    $taskWorker = Get-OwnedAppWorker
    if ($null -ne $taskWorker) {
        Stop-Process -Id $taskWorker.Id -ErrorAction Stop
        $taskWorker.WaitForExit(5000) | Out-Null
        if (!$taskWorker.HasExited) { throw 'Owned application worker did not stop.' }
    }
    Stop-OwnedProcess $taskAppStatePath
}

function Get-AuthSummary {
    $taskFiles = @(Get-ChildItem -LiteralPath $taskAuthDir -Filter '*.json' -File)
    $taskBound = 0
    foreach ($taskFile in $taskFiles) {
        Assert-NoReparsePath $taskFile.FullName
        $taskAuth = Get-Content -LiteralPath $taskFile.FullName -Raw | ConvertFrom-Json
        if (!$taskAuth.PSObject.Properties['type'] -or $taskAuth.type -ne 'codex') { throw 'Unexpected credential type in isolated auth directory.' }
        if (!$taskAuth.PSObject.Properties['use_siwc'] -or $taskAuth.use_siwc -ne $true -or
            !$taskAuth.PSObject.Properties['siwc'] -or $null -eq $taskAuth.siwc) {
            continue
        }
        if (!$taskAuth.siwc.PSObject.Properties['access_token'] -or [string]::IsNullOrWhiteSpace($taskAuth.siwc.access_token) -or
            !$taskAuth.siwc.PSObject.Properties['refresh_token'] -or [string]::IsNullOrWhiteSpace($taskAuth.siwc.refresh_token)) {
            throw 'SIWC metadata is incomplete; finish authorization before starting.'
        }
        if (!$taskAuth.siwc.PSObject.Properties['scope'] -or
            'chatgpt.tokens.use.direct' -notin ($taskAuth.siwc.scope -split '\s+') -or
            !$taskAuth.siwc.PSObject.Properties['client_id'] -or
            [string]::IsNullOrWhiteSpace($taskAuth.siwc.client_id) -or $taskAuth.siwc.client_id -eq 'dynamic_agent_client') {
            throw 'SIWC direct scope/issued client ID is missing; do not treat an identity-only login as plan authorization.'
        }
        $taskBound++
    }
    return @{ total = $taskFiles.Count; bound = $taskBound }
}

function Start-OwnedProcess([string[]]$Arguments, [string]$StatePath, [string]$LogPrefix) {
    if ($null -ne (Get-OwnedProcess $StatePath)) { throw 'This gateway action is already running.' }
    $taskProcess = Start-Process -FilePath $taskExe -ArgumentList $Arguments -WorkingDirectory $taskRuntime -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskLogDir "$LogPrefix.stdout.log") -RedirectStandardError (Join-Path $taskLogDir "$LogPrefix.stderr.log")
    $taskState = @{ owner = $taskOwner; runtime = $taskRuntime; executable = $taskExe; pid = $taskProcess.Id; started_ticks = $taskProcess.StartTime.ToUniversalTime().Ticks }
    Write-PrivateText $StatePath ($taskState | ConvertTo-Json)
    return $taskProcess
}

function Invoke-LocalRequest([string]$Path, [bool]$Authenticated) {
    $taskHeaders = @{}
    if ($Authenticated) { $taskHeaders.Authorization = 'Bearer ' + (Get-Content -LiteralPath $taskKeyPath -Raw).Trim() }
    try {
        $taskResponse = Invoke-WebRequest -UseBasicParsing -Uri ($taskBaseUrl + $Path) -Headers $taskHeaders -MaximumRedirection 0 -TimeoutSec 5
        return @{ status = [int]$taskResponse.StatusCode; content = $taskResponse.Content }
    } catch {
        if ($null -ne $_.Exception.Response) { return @{ status = [int]$_.Exception.Response.StatusCode; content = '' } }
        throw 'Loopback gateway is unavailable (details withheld to avoid leaking request headers).'
    } finally { $taskHeaders.Clear() }
}

if ($env:OS -ne 'Windows_NT' -or ![Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -notin @('AMD64', 'x86')) {
    throw 'This pinned launcher supports Windows x64 only.'
}
Assert-NoReparsePath $taskRuntime

if ($Action -eq 'Setup') {
    if (Test-Path -LiteralPath $taskRuntime) {
        Assert-Installation
        Write-Output "Verified existing CLIProxyAPI $taskVersion; fixed client key unchanged."
        return
    }
    New-Item -ItemType Directory -Path $taskRuntime -Force | Out-Null
    Set-PrivateDirectoryAcl $taskRuntime
    foreach ($taskPath in @($taskAuthDir, $taskLogDir, (Join-Path $taskRuntime 'downloads'), (Split-Path -Parent $taskExe))) {
        New-Item -ItemType Directory -Path $taskPath -Force | Out-Null
        Set-PrivateDirectoryAcl $taskPath
    }
    $taskReleaseRoot = "https://github.com/router-for-me/CLIProxyAPI/releases/download/v$taskVersion"
    $taskArchive = Join-Path $taskRuntime "downloads\CLIProxyAPI_${taskVersion}_windows_amd64.zip"
    $taskChecksums = Join-Path $taskRuntime 'downloads\checksums.txt'
    Invoke-WebRequest -UseBasicParsing -Uri "$taskReleaseRoot/checksums.txt" -OutFile $taskChecksums
    if ((Get-FileHash -LiteralPath $taskChecksums -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskChecksumHash) { throw 'Release checksum file does not match pinned GitHub digest.' }
    $taskChecksumLine = @(Get-Content -LiteralPath $taskChecksums | Where-Object { $_ -match ('^' + $taskArchiveHash + '\s+\*?CLIProxyAPI_' + [regex]::Escape($taskVersion) + '_windows_amd64\.zip$') })
    if ($taskChecksumLine.Count -ne 1) { throw 'Pinned archive digest does not match the release checksum entry.' }
    Invoke-WebRequest -UseBasicParsing -Uri "$taskReleaseRoot/CLIProxyAPI_${taskVersion}_windows_amd64.zip" -OutFile $taskArchive
    if ((Get-FileHash -LiteralPath $taskArchive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskArchiveHash) { throw 'Gateway archive SHA256 mismatch; refusing extraction.' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $taskZip = [IO.Compression.ZipFile]::OpenRead($taskArchive)
    try {
        foreach ($taskEntry in $taskZip.Entries) {
            $taskEntryPath = [IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $taskExe) $taskEntry.FullName))
            if (!$taskEntryPath.StartsWith((Split-Path -Parent $taskExe) + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe archive path.' }
        }
    } finally { $taskZip.Dispose() }
    Expand-Archive -LiteralPath $taskArchive -DestinationPath (Split-Path -Parent $taskExe)
    if (!(Test-Path -LiteralPath $taskExe -PathType Leaf)) { throw 'Pinned release did not contain the expected executable.' }
    $taskHelp = & $taskExe --help 2>&1 | Out-String
    if ($taskHelp -notmatch '8\.0\.33' -or $taskHelp -notmatch 'siwc-bind' -or $taskHelp -notmatch 'codex-login') { throw 'Executable help/version does not match expected SIWC capabilities.' }
    $taskRandom = New-Object byte[] 32
    $taskRng = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $taskRng.GetBytes($taskRandom) } finally { $taskRng.Dispose() }
    $taskClientKey = 'lg-local-' + [Convert]::ToBase64String($taskRandom).TrimEnd('=').Replace('+', '-').Replace('/', '_')
    Write-PrivateText $taskKeyPath $taskClientKey
    $taskYamlAuthDir = $taskAuthDir.Replace('\', '/')
    $taskYaml = @"
config-version: 8
server:
  host: "127.0.0.1"
  port: 8317
  max-request-body-bytes: 4194304
  trusted-proxies: []
  discovery:
    enabled: false
management:
  allow-remote: false
  secret-key: ""
  disable-control-panel: true
  disable-auto-update-panel: true
access:
  api-keys:
    - "$taskClientKey"
oauth:
  auth-dir: "$taskYamlAuthDir"
  auth-auto-refresh-workers: 1
routing:
  retry:
    request-retry: 0
    max-retry-credentials: 1
    max-retry-interval: 0
observability:
  logs:
    debug: false
    logging-to-file: false
    request-log: false
    error-logs-max-files: 1
  usage:
    usage-statistics-enabled: false
  pprof:
    enable: false
plugins:
  enabled: false
"@
    Write-PrivateText $taskConfig $taskYaml
    $taskManifest = @{ owner = $taskOwner; runtime = $taskRuntime; version = $taskVersion; source_sha = '2422c6b0f5fe16495320fa29456493874c1977a0'; archive_sha256 = $taskArchiveHash; exe_sha256 = (Get-FileHash -LiteralPath $taskExe -Algorithm SHA256).Hash; config_sha256 = (Get-FileHash -LiteralPath $taskConfig -Algorithm SHA256).Hash; key_sha256 = (Get-FileHash -LiteralPath $taskKeyPath -Algorithm SHA256).Hash }
    Write-PrivateText $taskManifestPath ($taskManifest | ConvertTo-Json)
    Remove-Variable taskClientKey, taskRandom, taskYaml -ErrorAction SilentlyContinue
    Assert-Installation
    Write-Output "Prepared verified CLIProxyAPI $taskVersion (Windows x64), private ACL, loopback-only config and fixed client key."
    Write-Output "Runtime: $taskRuntime"
    return
}

Assert-Installation
if ($Action -eq 'Stop') {
    Stop-OwnedProcess $taskStatePath
    Stop-OwnedProcess $taskOAuthStatePath
    Write-Output 'Owned gateway/OAuth processes stopped; credentials and client key retained.'
    return
}

if ($Action -eq 'StopLoreGuard') {
    Stop-OwnedApp
    Write-Output 'Owned loopback LoreGuard instance stopped; isolated data and credentials retained.'
    return
}

if ($Action -eq 'StartLoreGuard') {
    if (!(Test-Path -LiteralPath $taskAppPython -PathType Leaf)) { throw 'Repository .venv Python is required; no global Python will be selected.' }
    if ($null -ne (Get-OwnedProcess $taskAppStatePath)) {
        Write-Output 'Owned LoreGuard instance is already running at http://127.0.0.1:8787/.'
        return
    }
    if (@(Get-NetTCPConnection -State Listen -LocalPort 8787 -ErrorAction SilentlyContinue).Count -gt 0) { throw 'Port 8787 is occupied; no foreign process will be stopped.' }
    Assert-NoReparsePath $taskAppData
    if (!(Test-Path -LiteralPath $taskAppData)) { New-Item -ItemType Directory -Path $taskAppData | Out-Null }
    Set-PrivateDirectoryAcl $taskAppData
    $taskPreviousRuntime = [Environment]::GetEnvironmentVariable('LOREGUARD_PRO_GATEWAY_RUNTIME', 'Process')
    try {
        [Environment]::SetEnvironmentVariable('LOREGUARD_PRO_GATEWAY_RUNTIME', $taskRuntime, 'Process')
        $taskAppArgs = @('-m', 'uvicorn', 'scripts.local_pro_gateway_app:create_app', '--factory', '--app-dir', ('"' + $taskRepository + '"'), '--host', '127.0.0.1', '--port', '8787', '--no-proxy-headers', '--no-access-log')
        $taskAppProcess = Start-Process -FilePath $taskAppPython -ArgumentList $taskAppArgs -WorkingDirectory $taskRuntime -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskLogDir 'loreguard.stdout.log') -RedirectStandardError (Join-Path $taskLogDir 'loreguard.stderr.log')
    } finally { [Environment]::SetEnvironmentVariable('LOREGUARD_PRO_GATEWAY_RUNTIME', $taskPreviousRuntime, 'Process') }
    $taskAppState = @{ owner = $taskOwner; runtime = $taskRuntime; executable = $taskAppPython; repository = $taskRepository; pid = $taskAppProcess.Id; started_ticks = $taskAppProcess.StartTime.ToUniversalTime().Ticks }
    Write-PrivateText $taskAppStatePath ($taskAppState | ConvertTo-Json)
    $taskReady = $false
    for ($taskAttempt = 0; $taskAttempt -lt 40; $taskAttempt++) {
        try {
            $taskHealth = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8787/health' -MaximumRedirection 0 -TimeoutSec 2
            if ($taskHealth.StatusCode -eq 200 -and $taskHealth.Headers['X-LoreGuard-Local-Pro'] -eq '1') { $taskReady = $true; break }
        } catch { }
        Start-Sleep -Milliseconds 250
    }
    if (!$taskReady) { throw 'Isolated LoreGuard did not become healthy; inspect its private local logs.' }
    $taskWorker = Get-OwnedAppWorker
    $taskListeners = @(Get-NetTCPConnection -State Listen -LocalPort 8787 -ErrorAction Stop)
    if ($null -eq $taskWorker -or $taskListeners.Count -ne 1 -or $taskListeners[0].LocalAddress -ne '127.0.0.1' -or
        $taskListeners[0].OwningProcess -ne $taskWorker.Id) { throw 'Application listener does not belong to the verified local worker.' }
    Write-Output 'Isolated LoreGuard is running at http://127.0.0.1:8787/ (required account login; no shared/global model key).'
    return
}

if ($Action -eq 'Start') {
    if ($null -ne (Get-OwnedProcess $taskOAuthStatePath)) { throw 'OAuth is in progress; complete SIWC binding first.' }
    $taskAuthSummary = Get-AuthSummary
    if ($taskAuthSummary.total -ne $taskAuthSummary.bound) { throw 'Unbound Codex credentials found. Run SiwcBind before starting; backend-api fallback is not allowed.' }
    $taskExisting = Get-OwnedProcess $taskStatePath
    if ($null -eq $taskExisting) {
        if (@(Get-NetTCPConnection -State Listen -LocalPort 8317 -ErrorAction SilentlyContinue).Count -gt 0) { throw 'Port 8317 is occupied; no foreign process will be stopped.' }
        Start-OwnedProcess @('--config', ('"' + $taskConfig + '"'), '--local-model') $taskStatePath 'gateway' | Out-Null
    }
    $taskReady = $false
    for ($taskAttempt = 0; $taskAttempt -lt 20; $taskAttempt++) {
        try { $taskResponse = Invoke-LocalRequest '/models' $false; if ($taskResponse.status -eq 401) { $taskReady = $true; break } } catch { }
        Start-Sleep -Milliseconds 250
    }
    if (!$taskReady) { throw 'Gateway did not reach its authenticated loopback endpoint. Inspect private logs locally.' }
    Write-Output "Gateway is running at $taskBaseUrl; unauthenticated /models returns 401; SIWC credentials: $($taskAuthSummary.bound)."
    return
}

if ($Action -eq 'CodexLogin' -or $Action -eq 'SiwcBind') {
    Stop-OwnedProcess $taskStatePath
    if ($null -ne (Get-OwnedProcess $taskOAuthStatePath)) { throw 'An owned authorization attempt is already active.' }
    # v8.0.33 legacy Codex GenerateAuthURL hardcodes localhost:1455 even when
    # CallbackPort changes the listener. Only SIWC constructs a matching URI.
    $taskCallbackPort = 1455
    if ($Action -eq 'SiwcBind') { $taskCallbackPort = 1457 }
    if (@(Get-NetTCPConnection -State Listen -LocalPort $taskCallbackPort -ErrorAction SilentlyContinue).Count -gt 0) {
        throw "OAuth callback port $taskCallbackPort is occupied; no foreign process will be stopped."
    }
    $taskArgs = @('--config', ('"' + $taskConfig + '"'), '--local-model')
    if ($Action -eq 'CodexLogin') {
        $taskArgs += '--codex-login'
    } else {
        if ([string]::IsNullOrWhiteSpace($CredentialFileName) -or $CredentialFileName -notmatch '^[^\\/:*?"<>|]+\.json$' -or $CredentialFileName -in @('.', '..')) {
            throw 'SiwcBind requires a bare JSON filename under the isolated auth directory.'
        }
        $taskTarget = Join-Path $taskAuthDir $CredentialFileName
        if (!(Test-Path -LiteralPath $taskTarget -PathType Leaf)) { throw 'Credential file does not exist in this gateway auth directory.' }
        Assert-NoReparsePath $taskTarget
        $taskArgs += @('--oauth-callback-port', '1457', '--siwc-bind', ('"' + $CredentialFileName + '"'))
    }
    Start-OwnedProcess $taskArgs $taskOAuthStatePath 'oauth' | Out-Null
    Write-Output "$Action started; browser callback uses local port $taskCallbackPort. Complete only official browser authentication/consent; do not copy authorization URLs or credentials into chat."
    return
}

if ($Action -eq 'Models') {
    $taskAuthSummary = Get-AuthSummary
    if ($taskAuthSummary.bound -lt 1 -or $taskAuthSummary.bound -ne $taskAuthSummary.total) { throw 'No complete SIWC-bound account. Model listing is not an entitlement check before authorization.' }
    $taskResponse = Invoke-LocalRequest '/models' $true
    if ($taskResponse.status -ne 200) { throw "Gateway model catalog returned HTTP $($taskResponse.status)." }
    $taskModels = $taskResponse.content | ConvertFrom-Json
    @($taskModels.data) | ForEach-Object { Write-Output $_.id }
    return
}

$taskGatewayProcess = Get-OwnedProcess $taskStatePath
$taskOAuthProcess = Get-OwnedProcess $taskOAuthStatePath
$taskAppProcess = Get-OwnedProcess $taskAppStatePath
$taskAuthSummary = Get-AuthSummary
Write-Output ([pscustomobject]@{ Version = $taskVersion; BaseUrl = $taskBaseUrl; GatewayRunning = ($null -ne $taskGatewayProcess); OAuthRunning = ($null -ne $taskOAuthProcess); LoreGuardRunning = ($null -ne $taskAppProcess); LoreGuardUrl = 'http://127.0.0.1:8787/'; CredentialCount = $taskAuthSummary.total; SiwcBoundCount = $taskAuthSummary.bound; ClientKeyFile = $taskKeyPath; Runtime = $taskRuntime } | Format-List | Out-String)
