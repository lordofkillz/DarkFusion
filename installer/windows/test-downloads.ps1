# Run with Windows PowerShell 5.1; exercises the production download functions.
$ErrorActionPreference = 'Stop'
# Setup must not depend on module discovery in another application's environment.
function Get-FileHash { throw 'The downloader must use the built-in .NET checksum API.' }
Add-Type -AssemblyName System.Net.Http
Add-Type -ReferencedAssemblies @('System.Net.Http', 'System.Core') -TypeDefinition @'
using System;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
public class DownloadFixtureHandler : HttpMessageHandler {
 public byte[] Bytes;
 public bool IgnoreRange;
 public bool FirstDamaged;
 public int Calls;
 public long FirstRange = -1;
 protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellation) {
  Calls++;
  long start=0;
  if (request.Headers.Range!=null) foreach(var range in request.Headers.Range.Ranges) { start=range.From ?? 0; break; }
  if(Calls==1) FirstRange=start;
  bool partial=start>0 && !IgnoreRange;
  if(!partial) start=0;
  var body=new byte[Bytes.Length-start];
  Buffer.BlockCopy(Bytes,(int)start,body,0,body.Length);
  if(FirstDamaged && Calls==1) body[0]^=255;
  var result=new HttpResponseMessage(partial?HttpStatusCode.PartialContent:HttpStatusCode.OK);
  result.Content=new ByteArrayContent(body);
  if(partial) result.Content.Headers.ContentRange=new System.Net.Http.Headers.ContentRangeHeaderValue(start,Bytes.Length-1,Bytes.Length);
  return Task.FromResult(result);
 }
}
'@
$tokens=$null; $errors=$null
$tree=[Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'install-online.ps1'),[ref]$tokens,[ref]$errors)
if($errors.Count) { throw 'Download script does not parse' }
foreach($function in $tree.FindAll({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst]},$false)) {
 . ([ScriptBlock]::Create($function.Extent.Text))
}
$fixtureRoot=Join-Path ([IO.Path]::GetTempPath()) ('download-fixtures-'+[Guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($fixtureRoot)
$LogPath=Join-Path $fixtureRoot 'download.log'
$bytes=New-Object byte[] 1048576
(New-Object Random(17)).NextBytes($bytes)
$hasher=[Security.Cryptography.SHA256]::Create()
$digest=[BitConverter]::ToString($hasher.ComputeHash($bytes)).Replace('-','').ToLowerInvariant()
$hasher.Dispose()
$part=[pscustomobject]@{name='DarkFusion-runtime.001.bin';size_bytes=$bytes.Length;sha256=$digest;url='https://github.com/lordofkillz/DarkFusion/releases/download/test/fixture.bin'}
$passed=@()
foreach($scenario in @('fresh','resume','server_ignores_range','damaged_then_retry','cached','complete_partial')) {
 $cache=Join-Path $fixtureRoot $scenario
 [void][IO.Directory]::CreateDirectory($cache)
 $target=Join-Path $cache $part.name
 $handler=New-Object DownloadFixtureHandler
 $handler.Bytes=$bytes
 $handler.IgnoreRange=$scenario -eq 'server_ignores_range'
 $handler.FirstDamaged=$scenario -eq 'damaged_then_retry'
 if($scenario -in @('resume','server_ignores_range')) { [IO.File]::WriteAllBytes($target+'.partial',$bytes[0..4095]) }
 if($scenario -eq 'cached') { [IO.File]::WriteAllBytes($target,$bytes) }
 if($scenario -eq 'complete_partial') { [IO.File]::WriteAllBytes($target+'.partial',$bytes) }
 $client=New-Object Net.Http.HttpClient($handler)
 try {
  $received=Receive-Part $part 1 1
  if($received -cne $target -or -not (Test-Download $target $bytes.Length $digest)) { throw "Downloaded bytes failed: $scenario" }
  if($scenario -eq 'resume' -and $handler.FirstRange -ne 4096) { throw 'Resume offset was not requested' }
  if($scenario -in @('cached','complete_partial') -and $handler.Calls -ne 0) { throw 'Valid cached file was downloaded again' }
  if($scenario -eq 'damaged_then_retry' -and $handler.Calls -ne 2) { throw 'Damaged download was not retried' }
  $passed+=$scenario
  Write-Host "PASS: $scenario"
 } finally { $client.Dispose() }
}
Write-Host "All $($passed.Count) download checks passed. Fixtures: $fixtureRoot"
