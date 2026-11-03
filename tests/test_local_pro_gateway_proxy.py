"""Pure/mocked Windows launcher checks: never start the real gateway or read auth."""

from __future__ import annotations

import base64
from pathlib import Path
import shutil
import subprocess

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]


def test_local_gateway_proxy_policy_and_scoped_environment():
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        pytest.skip("Windows PowerShell 5.1 launcher policy")
    helper = REPOSITORY / "scripts" / "local-pro-gateway-proxy.ps1"
    script = r'''
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. '__HELPER__'
function Assert-True([bool]$Value, [string]$Name) {
    if (!$Value) { throw "Assertion failed: $Name" }
}
function Assert-Rejected([scriptblock]$Operation, [string]$Name) {
    $rejected = $false
    try { & $Operation | Out-Null } catch { $rejected = $true }
    Assert-True $rejected $Name
}
$enabled = [pscustomobject]@{ ProxyEnable = 1; ProxyServer = '127.0.0.1:7897' }
$disabled = [pscustomobject]@{ ProxyEnable = 0 }
Assert-True ((Resolve-LocalGatewayProxy $enabled @{}) -eq 'http://127.0.0.1:7897') 'manual proxy'
Assert-True ($null -eq (Resolve-LocalGatewayProxy $disabled @{})) 'disabled proxy'
Assert-True ($null -eq (Resolve-LocalGatewayProxy $null @{})) 'missing registry'
Assert-True ((Resolve-LocalGatewayProxy $enabled @{HTTP_PROXY='http://127.0.0.1:7897'; NO_PROXY='localhost,127.0.0.1,::1'}) -eq 'http://127.0.0.1:7897') 'matching env'
Assert-True ((ConvertTo-LocalGatewayProxyUri 'http://LOCALHOST:8080/') -eq 'http://localhost:8080') 'canonical localhost'
foreach ($value in @('http://remote.invalid:7897','http://user:private@127.0.0.1:7897','socks5://127.0.0.1:7897','https://127.0.0.1:7897', '127.0.0.1:0','127.0.0.1:65536','127.0.0.1:7897/path','127.0.0.1:7897?private','127.0.0.1:7897#private','127.0.0.1:7897 ','127.0.0.1:7897;https=127.0.0.1:7897','127.0.0.2:7897','http://localhost.:7897','http://2130706433:7897','http://[::1]:7897')) {
    Assert-Rejected { ConvertTo-LocalGatewayProxyUri $value } 'unsupported proxy'
}
Assert-Rejected { Resolve-LocalGatewayProxy $enabled @{HTTPS_PROXY='http://127.0.0.1:8888'} } 'conflicting env'
Assert-Rejected { Resolve-LocalGatewayProxy $disabled @{HTTP_PROXY='http://127.0.0.1:7897'} } 'no enabled matching proxy'
Assert-Rejected { Resolve-LocalGatewayProxy $enabled @{ALL_PROXY='http://remote.invalid:8888'} } 'remote env'
Assert-Rejected { Resolve-LocalGatewayProxy $enabled @{NO_PROXY='*'} } 'wildcard bypass'
Assert-Rejected { Resolve-LocalGatewayProxy $enabled @{NO_PROXY='api.openai.com'} } 'upstream bypass'
Assert-Rejected { Resolve-LocalGatewayProxy ([pscustomobject]@{ProxyEnable=1;ProxyServer='127.0.0.1:7897';AutoConfigURL='http://pac.invalid/private'}) @{} } 'PAC'
Assert-Rejected { Resolve-LocalGatewayProxy ([pscustomobject]@{ProxyEnable=1;ProxyServer='127.0.0.1:7897';AutoDetect=1}) @{} } 'automatic discovery'
Assert-Rejected { Resolve-LocalGatewayProxy ([pscustomobject]@{ProxyEnable=1}) @{} } 'missing enabled server'

Add-Type -AssemblyName System.Net.Http
$handler = New-Object System.Net.Http.HttpClientHandler
try {
    Set-LocalGatewayHttpProxy $handler 'http://127.0.0.1:7897'
    Assert-True $handler.UseProxy 'explicit model catalog proxy'
    Assert-True ($handler.Proxy.Address.AbsoluteUri -eq 'http://127.0.0.1:7897/') 'exact proxy'
    Assert-True ($null -eq $handler.Proxy.Credentials) 'no default credentials'
    Set-LocalGatewayHttpProxy $handler ''
    Assert-True (!$handler.UseProxy) 'explicit direct path'
} finally { $handler.Dispose() }

# Override only the resolver; the operation models synchronous child creation.
# No actual process, registry lookup, credential read or network request occurs.
function Get-LocalGatewayProxy { return 'http://127.0.0.1:7897' }
$original = @{}
foreach ($name in @('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY')) {
    $original[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    [Environment]::SetEnvironmentVariable($name, "original-$name", 'Process')
}
try {
    $result = Invoke-WithLocalGatewayProxy {
        Assert-True ($env:HTTP_PROXY -eq 'http://127.0.0.1:7897') 'child HTTP proxy'
        Assert-True ($env:HTTPS_PROXY -eq 'http://127.0.0.1:7897') 'child HTTPS proxy'
        Assert-True ([string]::IsNullOrEmpty($env:ALL_PROXY)) 'no other proxy'
        Assert-True ($env:NO_PROXY -eq '127.0.0.1,localhost,::1') 'loopback bypass'
        return 'created'
    }
    Assert-True ($result -eq 'created') 'return process object'
    foreach ($name in $original.Keys) {
        Assert-True ([Environment]::GetEnvironmentVariable($name,'Process') -eq "original-$name") 'restored success'
    }
    Assert-Rejected { Invoke-WithLocalGatewayProxy { throw 'synthetic spawn failure' } } 'spawn failure'
    foreach ($name in $original.Keys) {
        Assert-True ([Environment]::GetEnvironmentVariable($name,'Process') -eq "original-$name") 'restored failure'
    }
    foreach ($name in $original.Keys) { [Environment]::SetEnvironmentVariable($name,$null,'Process') }
    Invoke-WithLocalGatewayProxy { } | Out-Null
    foreach ($name in $original.Keys) {
        Assert-True ($null -eq [Environment]::GetEnvironmentVariable($name,'Process')) 'restored absent'
    }
} finally {
    foreach ($name in $original.Keys) { [Environment]::SetEnvironmentVariable($name,$original[$name],'Process') }
}
Write-Output 'gateway proxy policy and environment restoration passed'
'''.replace("__HELPER__", str(helper).replace("'", "''"))
    result = subprocess.run(
        [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand",
         base64.b64encode(script.encode("utf-16-le")).decode("ascii")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "gateway proxy policy and environment restoration passed" in result.stdout


def test_launcher_proxy_bridge_preserves_private_startup_and_integrity_checks():
    source = (REPOSITORY / "scripts" / "local-pro-gateway.ps1").read_text(encoding="utf-8")
    assert "local-pro-gateway-proxy.ps1" in source
    assert "$taskProcess = Invoke-WithLocalGatewayProxy {" in source
    assert "-WindowStyle Hidden -PassThru -RedirectStandardOutput" in source
    assert '-RedirectStandardError (Join-Path $taskLogDir "$LogPrefix.stderr.log")' in source
    assert "Set-LocalGatewayHttpProxy $taskHandler $taskUpstreamProxy" in source
    assert "$taskManifest.config_sha256" in source
    assert "$taskManifest.key_sha256" in source
    assert "Assert-Installation" in source

    # Static contract only: never execute a probe against an installed gateway.
    local_probe = source.split("function Invoke-LocalRequest", 1)[1].split(
        "if ($env:OS", 1
    )[0]
    assert "$taskHandler.UseProxy = $false" in local_probe
    assert "$taskHandler.AllowAutoRedirect = $false" in local_probe
    assert "$taskClient.Timeout = [TimeSpan]::FromSeconds(5)" in local_probe
    assert "'http://127.0.0.1:8317/v1/models'" in local_probe
    assert "$taskRequest.Headers.Authorization = $null" in local_probe
    assert "$taskResponse.Dispose()" in local_probe
    assert "$taskClient.Dispose()" in local_probe
    assert "$_" not in local_probe  # Never echo underlying request/headers.
