# Only bridge the current user's existing manual loopback HTTP proxy. This helper
# never writes registry values, user/machine environment, or gateway configuration.

function ConvertTo-LocalGatewayProxyUri([string]$Value) {
    if ($Value -notmatch '^(?i)(?:http://)?(127\.0\.0\.1|localhost):([1-9][0-9]{0,4})/?$') {
        throw 'Local gateway supports only a manual loopback HTTP proxy with an explicit port and no credentials.'
    }
    $taskProxyHost = $Matches[1].ToLowerInvariant()
    $taskProxyPort = [int]$Matches[2]
    if ($taskProxyPort -gt 65535) { throw 'Local gateway proxy port is invalid.' }
    return "http://${taskProxyHost}:$taskProxyPort"
}

function Resolve-LocalGatewayProxy([object]$InternetSettings, [hashtable]$ProxyEnvironment) {
    $taskProxy = $null
    if ($null -ne $InternetSettings) {
        if (($InternetSettings.PSObject.Properties['AutoConfigURL'] -and
             ![string]::IsNullOrWhiteSpace([string]$InternetSettings.AutoConfigURL)) -or
            ($InternetSettings.PSObject.Properties['AutoDetect'] -and $InternetSettings.AutoDetect -eq 1)) {
            throw 'Automatic/PAC proxy configuration is not supported by the local gateway launcher; no network settings were changed.'
        }
        if ($InternetSettings.PSObject.Properties['ProxyEnable'] -and $InternetSettings.ProxyEnable -eq 1) {
            if (!$InternetSettings.PSObject.Properties['ProxyServer']) {
                throw 'Enabled Windows proxy has no manual server; no network settings were changed.'
            }
            $taskProxy = ConvertTo-LocalGatewayProxyUri ([string]$InternetSettings.ProxyServer)
        }
    }
    foreach ($taskName in @('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY')) {
        $taskExisting = [string]$ProxyEnvironment[$taskName]
        if (![string]::IsNullOrWhiteSpace($taskExisting)) {
            $taskNormalized = ConvertTo-LocalGatewayProxyUri $taskExisting
            if ($null -eq $taskProxy -or $taskNormalized -ne $taskProxy) {
                throw 'Proxy environment conflicts with the current manual Windows proxy; no network settings were changed.'
            }
        }
    }
    if (![string]::IsNullOrWhiteSpace([string]$ProxyEnvironment['NO_PROXY'])) {
        foreach ($taskBypass in ([string]$ProxyEnvironment['NO_PROXY'] -split ',')) {
            if ($taskBypass.Trim() -notin @('127.0.0.1', 'localhost', '::1')) {
                throw 'Proxy bypass environment conflicts with the local-only gateway policy; no network settings were changed.'
            }
        }
    }
    return $taskProxy
}

function Get-LocalGatewayProxy {
    $taskSettings = Get-ItemProperty -LiteralPath 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' -ErrorAction SilentlyContinue
    $taskEnvironment = @{}
    foreach ($taskName in @('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY')) {
        $taskEnvironment[$taskName] = [Environment]::GetEnvironmentVariable($taskName, 'Process')
    }
    return Resolve-LocalGatewayProxy $taskSettings $taskEnvironment
}

function Invoke-WithLocalGatewayProxy([scriptblock]$Operation) {
    $taskProxy = Get-LocalGatewayProxy
    $taskOriginal = @{}
    $taskChildValues = @{ HTTP_PROXY = $taskProxy; HTTPS_PROXY = $taskProxy; ALL_PROXY = $null; NO_PROXY = '127.0.0.1,localhost,::1' }
    foreach ($taskName in $taskChildValues.Keys) {
        $taskOriginal[$taskName] = [Environment]::GetEnvironmentVariable($taskName, 'Process')
    }
    try {
        # Windows PowerShell 5.1 Start-Process has no -Environment parameter.
        # Its native file redirection outlives the launcher, unlike parent-owned
        # ProcessStartInfo pipe readers. Scope the inherited environment to this
        # synchronous spawn and restore every original value even on failure.
        foreach ($taskName in $taskChildValues.Keys) {
            [Environment]::SetEnvironmentVariable($taskName, $taskChildValues[$taskName], 'Process')
        }
        & $Operation
    } finally {
        foreach ($taskName in $taskOriginal.Keys) {
            [Environment]::SetEnvironmentVariable($taskName, $taskOriginal[$taskName], 'Process')
        }
    }
}

function Set-LocalGatewayHttpProxy([object]$Handler, [string]$Proxy) {
    $Handler.UseProxy = ![string]::IsNullOrWhiteSpace($Proxy)
    if ($Handler.UseProxy) {
        # Explicit proxy, no automatic discovery, default credentials or TLS
        # exceptions. The caller sends only to the fixed official HTTPS endpoint.
        $taskProxyUri = ConvertTo-LocalGatewayProxyUri $Proxy
        $Handler.Proxy = New-Object System.Net.WebProxy($taskProxyUri)
    }
}
