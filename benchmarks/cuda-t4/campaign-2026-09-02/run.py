"""The A100 half of the campaign: nine configurations, eleven rungs, two rounds.

Same ladder as 2026-07-25 -- the eleven targets cut from Darwin's Origin of
Species, Gutenberg #1228, the source benchmarks/prompts/cut_prompts.py uses.
A rung is a token count, so the ladder is cut per tokenizer.

The measurement is the campaign one: an OpenAI-compatible server, streaming,
temperature 0.8, 512 generated tokens, two rounds per rung, decode rate from the
stream's own token timings. What differs from the Radeon side is only the
machine and the stack, both of which every row records.

Checkpointed: a rung already in results.jsonl is not measured again, so a killed
session resumes instead of restarting.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import base64 as _b64, types as _types
_TSRC = _b64.b64decode(b'IiIiT25lIHRlbGVtZXRyeSBzaGFwZSBmb3IgZXZlcnkgbWFjaGluZSB0aGlzIHJlcG9zaXRvcnkgbWVhc3VyZXMgb24uCgpXaHkgdGhpcyBleGlzdHMKLS0tLS0tLS0tLS0tLS0tCkJlZm9yZSAyMDI2LTA5LTAyIHRoZSB0d28gcnVubmVyIGZhbWlsaWVzIHJlY29yZGVkIGRpc2pvaW50IHNldHMuIFRoZSBSYWRlb24KcnVubmVycyBzYW1wbGVkIHBlci1jYXJkIHBvd2VyIGFuZCBWUkFNIGZyb20gc3lzZnMgYW5kIHJlY29yZGVkIG5vIHdhbGwgY2xvY2s7CnRoZSBDVURBIHJ1bm5lcnMgcmVjb3JkZWQgYHdhbGxfc2AgYW5kIGBnZW5fdG9rZW5zYCBhbmQgc2FtcGxlZCBubyBoYXJkd2FyZSBhdAphbGwuIE5laXRoZXIgd2FzIGEgc3VwZXJzZXQgb2YgdGhlIG90aGVyLCBzbyBubyBxdWVzdGlvbiBjb3VsZCBiZSBhc2tlZCBhY3Jvc3MKbWFjaGluZXMgd2l0aG91dCBmaXJzdCBhc2tpbmcgd2hpY2ggY2FtcGFpZ24gaXQgY2FtZSBmcm9tLgoKV29yc2UsIG5laXRoZXIgc2FtcGxlZCBjbG9ja3MuIEEgQ29sYWIgVDQgbWVhc3VyZWQgb24gMjAyNi0wOS0wMiByYW4gYSA0MDAtc3RlcAptYXRtdWwgYXQgMTMwNSBNSHogYWdhaW5zdCBhIDE1OTAgTUh6IGNlaWxpbmcgd2hpbGUgZHJhd2luZyA3MS42IFcgYWdhaW5zdCBhCjcwIFcgY2FwIC0tIHBvd2VyLXRocm90dGxlZCB0aHJvdWdob3V0LiBOb3RoaW5nIGluIGVpdGhlciBzY2hlbWEgY291bGQgaGF2ZQpkaXN0aW5ndWlzaGVkIHRoYXQgZnJvbSBhIHNsb3cga2VybmVsLgoKV2hhdCBldmVyeSBmaWVsZCBoZXJlIGlzLCBhbmQgaXMgbm90Ci0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLQpFdmVyeSBmaWVsZCBiZWxvdyB3YXMgcHJvYmVkIG9uIHRoZSBhY3R1YWwgbWFjaGluZXMgb24gMjAyNi0wOS0wMiBhbmQgcmV0dXJucyBhCnJlYWwgdmFsdWUgb24gYXQgbGVhc3Qgb25lIHBsYXRmb3JtLiBGaWVsZHMgdGhhdCByZXR1cm5lZCBub3RoaW5nIGFueXdoZXJlIGFyZQphYnNlbnQgYnkgZGVjaXNpb24sIGFuZCBgQUJTRU5UYCByZWNvcmRzIHdoeSwgYmVjYXVzZSAid2UgZGlkIG5vdCByZWNvcmQgaXQiIGFuZAoiaXQgY2Fubm90IGJlIHJlY29yZGVkIiBhcmUgZGlmZmVyZW50IGZhY3RzIGFuZCB0aGUgbmV4dCBwZXJzb24gd2lsbCBhc2suCgogIGFjaGlldmVkIG1lbW9yeSBiYW5kd2lkdGgKICAgICAgTm90IGF2YWlsYWJsZSBvbiBnZngxMTAwIGhlcmUuIHJvY3Byb2Z2MyBydW5zIGluIHRoZSBWRklPIGd1ZXN0IGFuZCBpdHMKICAgICAga2VybmVsIHRyYWNlIGlzIGNvcnJlY3QgLS0gbmFtZXMsIGdyaWQgc2l6ZXMgYW5kIGR1cmF0aW9ucyBhbGwgbGFuZCAtLSBidXQKICAgICAgdGhlIG1lbW9yeS10cmFmZmljIGNvdW50ZXJzIHJlYWQgemVybzogRkVUQ0hfU0laRSwgR1JCTV9DT1VOVCwgR1BVQnVzeSBhbmQKICAgICAgU1FfSU5TVFNfVkFMVSBhbGwgc3VtbWVkIHRvIDAuMCBvdmVyIHNpeCBkaXNwYXRjaGVzIG9mIGEgMjA0OF4yIGZwMzIKICAgICAgbWF0bXVsLCB3aGlsZSBTUV9XQVZFUyBvbiB0aGUgc2FtZSBydW4gcmV0dXJuZWQgNjEyMC4gU29tZSBjb3VudGVyIGJsb2NrcwogICAgICBzdXJ2aXZlIHBhc3N0aHJvdWdoIGFuZCB0aGUgbWVtb3J5LXNpZGUgb25lcyBkbyBub3QuIE9uIENVREEgaXQgaXMKICAgICAgb2J0YWluYWJsZSwgYnV0IG9ubHkgdW5kZXIgYG5jdWAgb24gYSBtaWNyb2JlbmNobWFyazogYGRyYW1fX2J5dGVzLnN1bWAKICAgICAgcmV0dXJuZWQgMTguNzEgTUIgb24gYSBDb2xhYiBUNC4gSXQgaXMgbm90IG9idGFpbmFibGUgZHVyaW5nIGEgc2VydmVkIHJ1bgogICAgICBvbiBlaXRoZXIgcGxhdGZvcm0sIHdoaWNoIGlzIHdoYXQgdGhpcyBoYXJuZXNzIG1lYXN1cmVzLCBzbyB0aGUKICAgICAgdXRpbGlzYXRpb24gZmlndXJlcyB0aGlzIHJlcG9zaXRvcnkgcHVibGlzaGVzIHN0YXkgZGVyaXZlZCBmcm9tCiAgICAgIHRvay9zIHggYnl0ZXMvdG9rZW4gYW5kIGFyZSBub3QgY3Jvc3MtY2hlY2tlZCBhZ2FpbnN0IGhhcmR3YXJlLgoKICBQQ0llIHRocm91Z2hwdXQKICAgICAgTlZNTCBnaXZlcyBpdCAoYG52bWxEZXZpY2VHZXRQY2llVGhyb3VnaHB1dGAsIDM1MC8zMDAgS0IvcyBpZGxlIG9uIGEgVDQpLgogICAgICBhbWRncHUgZG9lcyBub3QgZXhwb3NlIGBwY2llX2J3YCBvbiBOYXZpIDMxIC0tIHRoZSBub2RlIGlzIGFic2VudCBvbiBib3RoCiAgICAgIGNhcmRzLiBSZWNvcmRlZCB3aGVyZSBpdCBleGlzdHMsIG51bGwgd2hlcmUgaXQgZG9lcyBub3QsIHJhdGhlciB0aGFuCiAgICAgIGRyb3BwZWQgZnJvbSB0aGUgc2NoZW1hLCBiZWNhdXNlIHRoZSBhc3ltbWV0cnkgaXMgYSBmYWN0IGFib3V0IHRoZQogICAgICBtYWNoaW5lcyBhbmQgaGlkaW5nIGl0IHdvdWxkIGludml0ZSBhIGNyb3NzLW1hY2hpbmUgY29tcGFyaXNvbiB0aGF0IGNhbm5vdAogICAgICBiZSBtYWRlLgoKICBhIHNlY29uZCBQTUMgY291bnRlciBpbiBvbmUgcGFzcwogICAgICByb2Nwcm9mdjMgYWJvcnRzIHdpdGggU0lHQUJSVCB3aGVuIGdpdmVuIHR3byBgLS1wbWNgIGNvdW50ZXJzIG9uIHRoaXMgYm94LAogICAgICBhdCBib3RoIDUxMl4yIGFuZCAyMDQ4XjIuIE9uZSBjb3VudGVyIHBlciBwYXNzIHdvcmtzIGF0IGJvdGggc2l6ZXMuIFRoZQogICAgICBheGlzIGlzIHRoZSBjb3VudGVyIGNvdW50LCBub3QgdGhlIHdvcmtsb2FkLgoKVXNhZ2UKLS0tLS0KICAgIGZyb20gaGFybmVzcy50ZWxlbWV0cnkgaW1wb3J0IFNhbXBsZXIsIGRlc2NyaWJlCgogICAgcyA9IFNhbXBsZXIoKTsgcy5zdGFydCgpCiAgICAuLi4gcnVuIG9uZSBjZWxsIC4uLgogICAgcm93LnVwZGF0ZShzLnN0b3BfYW5kX3N1bW1hcmlzZSgpKQogICAgbWV0YSA9IGRlc2NyaWJlKCkgICAgICAgICAgIyBvbmNlIHBlciBjb25maWd1cmF0aW9uCiIiIgoKZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoKaW1wb3J0IGdsb2IKaW1wb3J0IGpzb24KaW1wb3J0IG9zCmltcG9ydCByZQppbXBvcnQgdGhyZWFkaW5nCmltcG9ydCB0aW1lCgpTQ0hFTUFfVkVSU0lPTiA9IDEKCiM6IHdoYXQgdGhpcyBoYXJuZXNzIGRlbGliZXJhdGVseSBkb2VzIG5vdCBjYXJyeSwgYW5kIHdoeS4gS2VwdCBhcyBkYXRhIHNvIGEKIzogcmVhZGVyIGFuZCBhIGdhdGUgc2VlIHRoZSBzYW1lIGxpc3QuCkFCU0VOVCA9IHsKICAgICJhY2hpZXZlZF9tZW1fYndfZ2JzIjoKICAgICAgICAiZ2Z4MTEwMDogcm9jcHJvZnYzIG1lbW9yeSBjb3VudGVycyByZWFkIHplcm8gdW5kZXIgVkZJTyBwYXNzdGhyb3VnaCAiCiAgICAgICAgIihGRVRDSF9TSVpFIHN1bW1lZCAwLjAgd2hpbGUgU1FfV0FWRVMgcmV0dXJuZWQgNjEyMCkuIENVREE6IG9ubHkgdmlhICIKICAgICAgICAibmN1IG9uIGEgbWljcm9iZW5jaG1hcmssIG5vdCBkdXJpbmcgYSBzZXJ2ZWQgcnVuLiIsCiAgICAicGNpZV9id19yYWRlb24iOgogICAgICAgICJhbWRncHUgZXhwb3NlcyBubyBgcGNpZV9id2Agbm9kZSBvbiBOYXZpIDMxOyBOVk1MIGhhcyB0aGUgZXF1aXZhbGVudCwgIgogICAgICAgICJzbyB0aGUgZmllbGQgaXMgcHJlc2VudCBhbmQgbnVsbCBvbiB0aGUgUmFkZW9uIHNpZGUuIiwKfQoKCmRlZiBfZihwYXRoLCBjYXN0PWludCwgZGVmYXVsdD1Ob25lKToKICAgIHRyeToKICAgICAgICB3aXRoIG9wZW4ocGF0aCkgYXMgZmg6CiAgICAgICAgICAgIHJldHVybiBjYXN0KGZoLnJlYWQoKS5zdHJpcCgpKQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICByZXR1cm4gZGVmYXVsdAoKCmRlZiBfZHBtX2N1cnJlbnQocGF0aCk6CiAgICAiIiJUaGUgc3RhcnJlZCBlbnRyeSBvZiBhIHBwX2RwbV8qIHRhYmxlLCBpbiBNSHosIGFuZCB0aGUgdGFibGUncyBjZWlsaW5nLgoKICAgIFRoZSBmaWxlIGxvb2tzIGxpa2UgYDA6IDk2TWh6ICpcbjE6IDQ1Nk1oelxuLi4uYDsgdGhlIHN0YXIgaXMgdGhlIHN0YXRlIHRoZQogICAgY2FyZCBpcyBpbiByaWdodCBub3csIGFuZCB0aGUgbGFzdCByb3cgaXMgdGhlIGhpZ2hlc3Qgc3RhdGUgaXQgbWF5IGVudGVyLgogICAgIiIiCiAgICB0cnk6CiAgICAgICAgY3VyID0gdG9wID0gTm9uZQogICAgICAgIGZvciBsaW5lIGluIG9wZW4ocGF0aCk6CiAgICAgICAgICAgIG0gPSByZS5zZWFyY2gociIoXGQrKU1oeiIsIGxpbmUpCiAgICAgICAgICAgIGlmIG5vdCBtOgogICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgdiA9IGludChtLmdyb3VwKDEpKQogICAgICAgICAgICB0b3AgPSB2IGlmIHRvcCBpcyBOb25lIGVsc2UgbWF4KHRvcCwgdikKICAgICAgICAgICAgaWYgIioiIGluIGxpbmU6CiAgICAgICAgICAgICAgICBjdXIgPSB2CiAgICAgICAgcmV0dXJuIGN1ciwgdG9wCiAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgIHJldHVybiBOb25lLCBOb25lCgoKY2xhc3MgX0FtZENhcmQ6CiAgICAiIiJPbmUgcGFzc2VkLXRocm91Z2ggUmFkZW9uLCByZWFkIGVudGlyZWx5IGZyb20gc3lzZnMuIiIiCgogICAgZGVmIF9faW5pdF9fKHNlbGYsIGRldik6CiAgICAgICAgc2VsZi5kZXYgPSBkZXYKICAgICAgICBzZWxmLmh3bW9uID0gbmV4dChpdGVyKGdsb2IuZ2xvYihvcy5wYXRoLmpvaW4oZGV2LCAiaHdtb24iLCAiaHdtb24qIikpKSwgTm9uZSkKCiAgICBkZWYgc2xvdChzZWxmKToKICAgICAgICB0cnk6CiAgICAgICAgICAgIGZvciBsaW5lIGluIG9wZW4ob3MucGF0aC5qb2luKHNlbGYuZGV2LCAidWV2ZW50IikpOgogICAgICAgICAgICAgICAgaWYgbGluZS5zdGFydHN3aXRoKCJQQ0lfU0xPVF9OQU1FPSIpOgogICAgICAgICAgICAgICAgICAgIHJldHVybiBsaW5lLnN0cmlwKCkuc3BsaXQoIj0iLCAxKVsxXQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgIHBhc3MKICAgICAgICByZXR1cm4gTm9uZQoKICAgIGRlZiBzYW1wbGUoc2VsZik6CiAgICAgICAgcywgc190b3AgPSBfZHBtX2N1cnJlbnQob3MucGF0aC5qb2luKHNlbGYuZGV2LCAicHBfZHBtX3NjbGsiKSkKICAgICAgICBtLCBtX3RvcCA9IF9kcG1fY3VycmVudChvcy5wYXRoLmpvaW4oc2VsZi5kZXYsICJwcF9kcG1fbWNsayIpKQogICAgICAgIHIgPSB7CiAgICAgICAgICAgICJncHVfYnVzeV9wY3QiOiBfZihvcy5wYXRoLmpvaW4oc2VsZi5kZXYsICJncHVfYnVzeV9wZXJjZW50IikpLAogICAgICAgICAgICAibWVtX2J1c3lfcGN0IjogX2Yob3MucGF0aC5qb2luKHNlbGYuZGV2LCAibWVtX2J1c3lfcGVyY2VudCIpKSwKICAgICAgICAgICAgInZyYW1fdXNlZF9iIjogX2Yob3MucGF0aC5qb2luKHNlbGYuZGV2LCAibWVtX2luZm9fdnJhbV91c2VkIikpLAogICAgICAgICAgICAic2Nsa19taHoiOiBzLCAibWNsa19taHoiOiBtLAogICAgICAgICAgICAic2Nsa19taHpfY2FwIjogc190b3AsICJtY2xrX21oel9jYXAiOiBtX3RvcCwKICAgICAgICAgICAgInBjaWVfdHhfa2JzIjogTm9uZSwgInBjaWVfcnhfa2JzIjogTm9uZSwgICAjIG5vIHBjaWVfYncgb24gTmF2aSAzMQogICAgICAgIH0KICAgICAgICBpZiBzZWxmLmh3bW9uOgogICAgICAgICAgICByWyJwb3dlcl93Il0gPSAoX2Yob3MucGF0aC5qb2luKHNlbGYuaHdtb24sICJwb3dlcjFfYXZlcmFnZSIpLCBpbnQsIDApIG9yIDApIC8gMWU2CiAgICAgICAgICAgIHJbInBvd2VyX2NhcF93Il0gPSAoX2Yob3MucGF0aC5qb2luKHNlbGYuaHdtb24sICJwb3dlcjFfY2FwIiksIGludCwgMCkgb3IgMCkgLyAxZTYKICAgICAgICAgICAgclsidGVtcF9jIl0gPSAoX2Yob3MucGF0aC5qb2luKHNlbGYuaHdtb24sICJ0ZW1wMV9pbnB1dCIpLCBpbnQsIDApIG9yIDApIC8gMWUzCiAgICAgICAgcmV0dXJuIHIKCiAgICBkZWYgc3RhdGljKHNlbGYpOgogICAgICAgIHJldHVybiB7InNsb3QiOiBzZWxmLnNsb3QoKSwKICAgICAgICAgICAgICAgICJ2cmFtX3RvdGFsX2IiOiBfZihvcy5wYXRoLmpvaW4oc2VsZi5kZXYsICJtZW1faW5mb192cmFtX3RvdGFsIikpLAogICAgICAgICAgICAgICAgImxpbmtfc3BlZWQiOiBfZihvcy5wYXRoLmpvaW4oc2VsZi5kZXYsICJjdXJyZW50X2xpbmtfc3BlZWQiKSwgc3RyKSwKICAgICAgICAgICAgICAgICJsaW5rX3dpZHRoIjogX2Yob3MucGF0aC5qb2luKHNlbGYuZGV2LCAiY3VycmVudF9saW5rX3dpZHRoIiksIHN0cil9CgoKY2xhc3MgX052Q2FyZDoKICAgICIiIk9uZSBOVklESUEgZGV2aWNlLCByZWFkIHRocm91Z2ggTlZNTCBpbi1wcm9jZXNzLgoKICAgIE5WTUwgY29zdHMgOS4yIG1zIGZvciBhIGZ1bGwgc2FtcGxlIGFnYWluc3QgMjkuOCBtcyBmb3Igb25lIGBudmlkaWEtc21pYAogICAgc3VicHJvY2VzcywgbWVhc3VyZWQgb24gYSBDb2xhYiBUNCwgc28gdGhlIHNhbXBsZXIgbmV2ZXIgc2hlbGxzIG91dC4KICAgICIiIgoKICAgIGRlZiBfX2luaXRfXyhzZWxmLCBpZHgpOgogICAgICAgIGltcG9ydCBweW52bWwKICAgICAgICBzZWxmLlAgPSBweW52bWwKICAgICAgICBzZWxmLmggPSBweW52bWwubnZtbERldmljZUdldEhhbmRsZUJ5SW5kZXgoaWR4KQoKICAgIGRlZiBzYW1wbGUoc2VsZik6CiAgICAgICAgUCwgaCA9IHNlbGYuUCwgc2VsZi5oCiAgICAgICAgdSA9IFAubnZtbERldmljZUdldFV0aWxpemF0aW9uUmF0ZXMoaCkKICAgICAgICByID0geyJncHVfYnVzeV9wY3QiOiB1LmdwdSwgIm1lbV9idXN5X3BjdCI6IHUubWVtb3J5LAogICAgICAgICAgICAgInZyYW1fdXNlZF9iIjogUC5udm1sRGV2aWNlR2V0TWVtb3J5SW5mbyhoKS51c2VkLAogICAgICAgICAgICAgInBvd2VyX3ciOiBQLm52bWxEZXZpY2VHZXRQb3dlclVzYWdlKGgpIC8gMWUzLAogICAgICAgICAgICAgInNjbGtfbWh6IjogUC5udm1sRGV2aWNlR2V0Q2xvY2tJbmZvKGgsIFAuTlZNTF9DTE9DS19TTSksCiAgICAgICAgICAgICAibWNsa19taHoiOiBQLm52bWxEZXZpY2VHZXRDbG9ja0luZm8oaCwgUC5OVk1MX0NMT0NLX01FTSl9CiAgICAgICAgZm9yIGtleSwgZm4gaW4gKCgic2Nsa19taHpfY2FwIiwgbGFtYmRhOiBQLm52bWxEZXZpY2VHZXRNYXhDbG9ja0luZm8oaCwgUC5OVk1MX0NMT0NLX1NNKSksCiAgICAgICAgICAgICAgICAgICAgICAgICgibWNsa19taHpfY2FwIiwgbGFtYmRhOiBQLm52bWxEZXZpY2VHZXRNYXhDbG9ja0luZm8oaCwgUC5OVk1MX0NMT0NLX01FTSkpLAogICAgICAgICAgICAgICAgICAgICAgICAoInBvd2VyX2NhcF93IiwgbGFtYmRhOiBQLm52bWxEZXZpY2VHZXRFbmZvcmNlZFBvd2VyTGltaXQoaCkgLyAxZTMpLAogICAgICAgICAgICAgICAgICAgICAgICAoInRlbXBfYyIsIGxhbWJkYTogUC5udm1sRGV2aWNlR2V0VGVtcGVyYXR1cmUoaCwgUC5OVk1MX1RFTVBFUkFUVVJFX0dQVSkpLAogICAgICAgICAgICAgICAgICAgICAgICAoInBjaWVfdHhfa2JzIiwgbGFtYmRhOiBQLm52bWxEZXZpY2VHZXRQY2llVGhyb3VnaHB1dCgKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGgsIFAuTlZNTF9QQ0lFX1VUSUxfVFhfQllURVMpKSwKICAgICAgICAgICAgICAgICAgICAgICAgKCJwY2llX3J4X2ticyIsIGxhbWJkYTogUC5udm1sRGV2aWNlR2V0UGNpZVRocm91Z2hwdXQoCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBoLCBQLk5WTUxfUENJRV9VVElMX1JYX0JZVEVTKSkpOgogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICByW2tleV0gPSBmbigpCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICAgICByW2tleV0gPSBOb25lCiAgICAgICAgcmV0dXJuIHIKCiAgICBkZWYgc3RhdGljKHNlbGYpOgogICAgICAgIFAsIGggPSBzZWxmLlAsIHNlbGYuaAogICAgICAgIG91dCA9IHsic2xvdCI6IE5vbmUsICJ2cmFtX3RvdGFsX2IiOiBQLm52bWxEZXZpY2VHZXRNZW1vcnlJbmZvKGgpLnRvdGFsfQogICAgICAgIGZvciBrZXksIGZuIGluICgoImxpbmtfc3BlZWQiLCBsYW1iZGE6IHN0cihQLm52bWxEZXZpY2VHZXRDdXJyUGNpZUxpbmtHZW5lcmF0aW9uKGgpKSksCiAgICAgICAgICAgICAgICAgICAgICAgICgibGlua193aWR0aCIsIGxhbWJkYTogc3RyKFAubnZtbERldmljZUdldEN1cnJQY2llTGlua1dpZHRoKGgpKSkpOgogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICBvdXRba2V5XSA9IGZuKCkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIG91dFtrZXldID0gTm9uZQogICAgICAgIHJldHVybiBvdXQKCgpkZWYgY2FyZHMoKToKICAgICIiIkV2ZXJ5IGFjY2VsZXJhdG9yIHRoaXMgbWFjaGluZSB3aWxsIGJlIG1lYXN1cmVkIG9uLCBpbiBhIHN0YWJsZSBvcmRlci4iIiIKICAgIGRldnMgPSBzb3J0ZWQoZCBmb3IgZCBpbiBnbG9iLmdsb2IoIi9zeXMvY2xhc3MvZHJtL2NhcmQqL2RldmljZSIpCiAgICAgICAgICAgICAgICAgIGlmIG9zLnBhdGguZXhpc3RzKG9zLnBhdGguam9pbihkLCAiZ3B1X2J1c3lfcGVyY2VudCIpKSkKICAgIGlmIGRldnM6CiAgICAgICAgcmV0dXJuIFtfQW1kQ2FyZChkKSBmb3IgZCBpbiBkZXZzXQogICAgdHJ5OgogICAgICAgIGltcG9ydCBweW52bWwKICAgICAgICBweW52bWwubnZtbEluaXQoKQogICAgICAgIHJldHVybiBbX052Q2FyZChpKSBmb3IgaSBpbiByYW5nZShweW52bWwubnZtbERldmljZUdldENvdW50KCkpXQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICByZXR1cm4gW10KCgpjbGFzcyBTYW1wbGVyKHRocmVhZGluZy5UaHJlYWQpOgogICAgIiIiUG9sbHMgZXZlcnkgY2FyZCBvbiBhIGZpeGVkIGNhZGVuY2UgZm9yIHRoZSBsZW5ndGggb2Ygb25lIGNlbGwuCgogICAgVGhlIGZpcnN0IHRoaXJkIGFuZCB0aGUgbGFzdCBzaXh0aCBvZiB0aGUgc2FtcGxlcyBhcmUgZHJvcHBlZCwgYXMgdGhlIFJhZGVvbgogICAgcnVubmVycyBoYXZlIGFsd2F5cyBkb25lOiB0aGUgaGVhZCBpcyBlbmdpbmUgd2FybS11cCBhbmQgdGhlIHRhaWwgaXMgdGhlCiAgICByZXF1ZXN0IGRyYWluaW5nLCBhbmQgbmVpdGhlciBpcyB0aGUgc3RlYWR5IHN0YXRlIHRoZSBjZWxsIGlzIGFib3V0LgogICAgIiIiCgogICAgUEVSSU9EX1MgPSAxLjUKCiAgICBkZWYgX19pbml0X18oc2VsZiwgcGVyaW9kX3M9Tm9uZSk6CiAgICAgICAgc3VwZXIoKS5fX2luaXRfXyhkYWVtb249VHJ1ZSkKICAgICAgICBzZWxmLnBlcmlvZCA9IHBlcmlvZF9zIG9yIHNlbGYuUEVSSU9EX1MKICAgICAgICBzZWxmLnN0b3BfZXYgPSB0aHJlYWRpbmcuRXZlbnQoKQogICAgICAgIHNlbGYucm93cyA9IFtdCiAgICAgICAgc2VsZi5jYXJkcyA9IGNhcmRzKCkKCiAgICBkZWYgcnVuKHNlbGYpOgogICAgICAgIHdoaWxlIG5vdCBzZWxmLnN0b3BfZXYuaXNfc2V0KCk6CiAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgIHNlbGYucm93cy5hcHBlbmQoW2Muc2FtcGxlKCkgZm9yIGMgaW4gc2VsZi5jYXJkc10pCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICAgICBwYXNzCiAgICAgICAgICAgIHNlbGYuc3RvcF9ldi53YWl0KHNlbGYucGVyaW9kKQoKICAgIGRlZiBfX2VudGVyX18oc2VsZik6CiAgICAgICAgc2VsZi50MCA9IHRpbWUucGVyZl9jb3VudGVyKCkKICAgICAgICBzZWxmLnN0YXJ0KCkKICAgICAgICByZXR1cm4gc2VsZgoKICAgIGRlZiBfX2V4aXRfXyhzZWxmLCAqZXhjKToKICAgICAgICBzZWxmLnJlc3VsdCA9IHNlbGYuc3RvcF9hbmRfc3VtbWFyaXNlKCkKICAgICAgICBzZWxmLnJlc3VsdFsid2FsbF9zIl0gPSByb3VuZCh0aW1lLnBlcmZfY291bnRlcigpIC0gc2VsZi50MCwgMykKICAgICAgICByZXR1cm4gRmFsc2UKCiAgICBkZWYgc3RvcF9hbmRfc3VtbWFyaXNlKHNlbGYpOgogICAgICAgIHNlbGYuc3RvcF9ldi5zZXQoKQogICAgICAgIHNlbGYuam9pbih0aW1lb3V0PXNlbGYucGVyaW9kICogMiArIDEpCiAgICAgICAgcmV0dXJuIHN1bW1hcmlzZShzZWxmLnJvd3MpCgoKIzogdGhlIGFnZ3JlZ2F0ZSBrZXlzIGV2ZXJ5IG1lYXN1cmVkIHJvdyBjYXJyaWVzLCB3aGF0ZXZlciB0aGUgcGxhdGZvcm0gYW5kCiM6IGhvd2V2ZXIgc2hvcnQgdGhlIGNlbGwuIEEgNTAwLXRva2VuIHByZWZpbGwgY2FuIGZpbmlzaCBpbnNpZGUgb25lIHNhbXBsaW5nCiM6IHBlcmlvZDsgdGhlIHNoYXBlIG11c3Qgbm90IGNoYW5nZSBiZWNhdXNlIG9mIHRoYXQsIG9yIGhhbGYgdGhlIHJvd3MgZ3JvdyBhCiM6IGRpZmZlcmVudCBzY2hlbWEgYW5kIHRoZSBjb21wYXJpc29uIHF1aWV0bHkgYmVjb21lcyBjb25kaXRpb25hbC4KU1VNTUFSWV9LRVlTID0gKCJncHVfYnVzeV9wY3RfbWF4IiwgIm1lbV9idXN5X3BjdF9tYXgiLCAicG93ZXJfd19tYXgiLAogICAgICAgICAgICAgICAgInRlbXBfY19tYXgiLCAic2Nsa19taHpfbWF4IiwgIm1jbGtfbWh6X21heCIsICJ2cmFtX3VzZWRfYl9tYXgiLAogICAgICAgICAgICAgICAgInBjaWVfdHhfa2JzX21heCIsICJwY2llX3J4X2tic19tYXgiLCAicG93ZXJfd19zdW1fbWF4IiwKICAgICAgICAgICAgICAgICJwb3dlcl93X3N1bV9taW4iLCAic2Nsa19taHpfY2FwIiwgInBvd2VyX2NhcF93IiwKICAgICAgICAgICAgICAgICJzY2xrX3BjdF9vZl9jYXAiKQoKCmRlZiBfZW1wdHkoKToKICAgIGQgPSB7InRlbGVfc2FtcGxlcyI6IDAsICJ0ZWxlX3NjaGVtYSI6IFNDSEVNQV9WRVJTSU9OLCAicGVyX2NhcmQiOiB7fX0KICAgIGQudXBkYXRlKHtrOiBOb25lIGZvciBrIGluIFNVTU1BUllfS0VZU30pCiAgICByZXR1cm4gZAoKCmRlZiBzdW1tYXJpc2Uocm93cyk6CiAgICAiIiJQZXItY2VsbCBhZ2dyZWdhdGVzLCB3aXRoIHRoZSBmaWVsZCBuYW1lcyBldmVyeSBtYWNoaW5lIGVtaXRzLiIiIgogICAgaWYgbGVuKHJvd3MpID49IDY6CiAgICAgICAgcm93cyA9IHJvd3NbbGVuKHJvd3MpIC8vIDM6IC1tYXgoMSwgbGVuKHJvd3MpIC8vIDYpXQogICAgaWYgbm90IHJvd3M6CiAgICAgICAgcmV0dXJuIF9lbXB0eSgpCiAgICBuID0gbGVuKHJvd3NbMF0pCgogICAgZGVmIHZhbHMoY2FyZCwga2V5KToKICAgICAgICByZXR1cm4gW3JbY2FyZF0uZ2V0KGtleSkgZm9yIHIgaW4gcm93cyBpZiByW2NhcmRdLmdldChrZXkpIGlzIG5vdCBOb25lXQoKICAgIG91dCA9IHsidGVsZV9zYW1wbGVzIjogbGVuKHJvd3MpLCAidGVsZV9zY2hlbWEiOiBTQ0hFTUFfVkVSU0lPTn0KICAgIGFnZyA9IHt9CiAgICBmb3Iga2V5LCBob3cgaW4gKCgiZ3B1X2J1c3lfcGN0IiwgIm1heCIpLCAoIm1lbV9idXN5X3BjdCIsICJtYXgiKSwKICAgICAgICAgICAgICAgICAgICAgKCJwb3dlcl93IiwgIm1heCIpLCAoInRlbXBfYyIsICJtYXgiKSwKICAgICAgICAgICAgICAgICAgICAgKCJzY2xrX21oeiIsICJtYXgiKSwgKCJtY2xrX21oeiIsICJtYXgiKSwKICAgICAgICAgICAgICAgICAgICAgKCJ2cmFtX3VzZWRfYiIsICJtYXgiKSwKICAgICAgICAgICAgICAgICAgICAgKCJwY2llX3R4X2ticyIsICJtYXgiKSwgKCJwY2llX3J4X2ticyIsICJtYXgiKSk6CiAgICAgICAgcGVyID0gW10KICAgICAgICBmb3IgYyBpbiByYW5nZShuKToKICAgICAgICAgICAgdiA9IHZhbHMoYywga2V5KQogICAgICAgICAgICBwZXIuYXBwZW5kKG1heCh2KSBpZiB2IGVsc2UgTm9uZSkKICAgICAgICBhZ2dba2V5XSA9IHBlcgogICAgICAgIGdvdCA9IFt4IGZvciB4IGluIHBlciBpZiB4IGlzIG5vdCBOb25lXQogICAgICAgIG91dFtmIntrZXl9X21heCJdID0gbWF4KGdvdCkgaWYgZ290IGVsc2UgTm9uZQogICAgIyBwb3dlciBpcyB0aGUgb25lIHF1YW50aXR5IHRoYXQgaXMgc3VtbWVkIGFjcm9zcyBjYXJkcyByYXRoZXIgdGhhbiBtYXhlZDoKICAgICMgYSB0d28tY2FyZCBib3gncyBkcmF3IGlzIHdoYXQgdGhlIHdhbGwgc2Vlcy4KICAgIHBzdW0gPSBbc3VtKHJbY10uZ2V0KCJwb3dlcl93Iikgb3IgMCBmb3IgYyBpbiByYW5nZShuKSkgZm9yIHIgaW4gcm93c10KICAgIG91dFsicG93ZXJfd19zdW1fbWF4Il0gPSByb3VuZChtYXgocHN1bSksIDEpIGlmIHBzdW0gZWxzZSBOb25lCiAgICBvdXRbInBvd2VyX3dfc3VtX21pbiJdID0gcm91bmQobWluKHBzdW0pLCAxKSBpZiBwc3VtIGVsc2UgTm9uZQogICAgb3V0WyJwZXJfY2FyZCJdID0ge2s6IHYgZm9yIGssIHYgaW4gYWdnLml0ZW1zKCl9CiAgICAjIHRoZSBndWFyZCByYWlsIHRoZSBvbGQgc2NoZW1hcyBjb3VsZCBub3QgZXhwcmVzcwogICAgY2Fwc19zID0gW3Jvd3NbLTFdW2NdLmdldCgic2Nsa19taHpfY2FwIikgZm9yIGMgaW4gcmFuZ2UobildCiAgICBjYXBzX3AgPSBbcm93c1stMV1bY10uZ2V0KCJwb3dlcl9jYXBfdyIpIGZvciBjIGluIHJhbmdlKG4pXQogICAgb3V0WyJzY2xrX21oel9jYXAiXSA9IG1heChbYyBmb3IgYyBpbiBjYXBzX3MgaWYgY10sIGRlZmF1bHQ9Tm9uZSkKICAgIG91dFsicG93ZXJfY2FwX3ciXSA9IG1heChbYyBmb3IgYyBpbiBjYXBzX3AgaWYgY10sIGRlZmF1bHQ9Tm9uZSkKICAgIG91dFsic2Nsa19wY3Rfb2ZfY2FwIl0gPSAoCiAgICAgICAgcm91bmQob3V0WyJzY2xrX21oel9tYXgiXSAvIG91dFsic2Nsa19taHpfY2FwIl0gKiAxMDAsIDEpCiAgICAgICAgaWYgb3V0LmdldCgic2Nsa19taHpfbWF4IikgYW5kIG91dC5nZXQoInNjbGtfbWh6X2NhcCIpIGVsc2UgTm9uZSkKICAgIGZvciBrIGluIFNVTU1BUllfS0VZUzogICAgICAgICAgICAgICAgICAgICAgICAjIG5ldmVyIGEgcmFnZ2VkIHJvdwogICAgICAgIG91dC5zZXRkZWZhdWx0KGssIE5vbmUpCiAgICByZXR1cm4gb3V0CgoKZGVmIGRlc2NyaWJlKCk6CiAgICAiIiJPbmUtdGltZSBtYWNoaW5lIGRlc2NyaXB0aW9uLCB3cml0dGVuIGJlc2lkZSBldmVyeSBjYW1wYWlnbidzIHJvd3MuIiIiCiAgICBjcyA9IGNhcmRzKCkKICAgIHJldHVybiB7ImtpbmQiOiAidGVsZW1ldHJ5X21ldGEiLCAidGVsZV9zY2hlbWEiOiBTQ0hFTUFfVkVSU0lPTiwKICAgICAgICAgICAgIm5fY2FyZHMiOiBsZW4oY3MpLAogICAgICAgICAgICAiY2FyZHMiOiBbYy5zdGF0aWMoKSBmb3IgYyBpbiBjc10sCiAgICAgICAgICAgICJhYnNlbnQiOiBBQlNFTlQsCiAgICAgICAgICAgICJ0cyI6IHRpbWUudGltZSgpfQoKCmlmIF9fbmFtZV9fID09ICJfX21haW5fXyI6ICAgICAgICAgICAgICAgICAgICAgICAjIHNtb2tlIHRlc3Qgb24gYW55IG1hY2hpbmUKICAgIHMgPSBTYW1wbGVyKHBlcmlvZF9zPTAuMykKICAgIHMuc3RhcnQoKQogICAgdGltZS5zbGVlcCgxLjUpCiAgICBwcmludChqc29uLmR1bXBzKHsiZGVzY3JpYmUiOiBkZXNjcmliZSgpLCAic2FtcGxlIjogcy5zdG9wX2FuZF9zdW1tYXJpc2UoKX0sCiAgICAgICAgICAgICAgICAgICAgIGluZGVudD0xLCBkZWZhdWx0PXN0cikpCg==').decode()
_tm = _types.ModuleType("telemetry"); exec(compile(_TSRC, "telemetry.py", "exec"), _tm.__dict__)
Sampler, describe = _tm.Sampler, _tm.describe   # harness/telemetry.py, inlined for the VM

MACHINE = os.environ.get("BENCH_MACHINE", "T4")   # goes on every row
D = "/content/work"
RES = f"{D}/results.jsonl"
PROG = f"{D}/PROGRESS.txt"
MODELS = "/content/models"
PORT = 8000
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HEALTH = f"http://127.0.0.1:{PORT}/health"
TARGETS = [500, 8000, 32000]   # 2026-09-02: the 500 rung x5, the other two x2
GEN = 512
MML = 33000

# BENCH_CFGS picks a subset by id, as the Radeon runner does.
CFGS = [
    # 2026-08-30, the T4, second attempt. The first ran to 17 of 22 rungs and
    # was lost with its session, so nothing of it survives.
    #
    # float16 because Turing has no bf16; util 0.95 with mns 1 because vLLM
    # sizes activations and CUDA graphs for max_num_seqs and the default capture
    # set costs 4.57 GiB of a 15 GiB card. Needs vllm#39018 to start at all:
    # without it the engine dies at kernel load asking 98304 bytes of shared
    # memory against Turing's 65536. Every row records that patch.
    dict(id="G12", model="gemma-4-12B-it-qat-w4a16-ct",
         dtype="float16", util=0.95, mns=1),
]

# gemma-4 registers image, video and audio. vLLM only drops the mm-prefix
# backend requirement when every registered modality is zero, and without that
# FlashInfer is refused at engine init regardless of routing -- which is how the
# spec article's collapse gets measured by accident on this machine.
GEMMA_MM = '--limit-mm-per-prompt \'{"image":0,"video":0,"audio":0}\''


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(PROG, "a") as f:
        f.write(line + "\n")


def emit(obj):
    obj["ts"] = round(time.time(), 1)
    with open(RES, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def done_keys():
    ks = set()
    if os.path.exists(RES):
        for l in open(RES):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("kind") == "decode":
                ks.add((r["cfg"], r["target"], r["round"]))
            if r.get("kind") in ("config_complete", "config_failed"):
                ks.add(("cfg", r["cfg"]))
    return ks


# --- the ladder, cut per tokenizer -----------------------------------------
BOOK = "/content/work/origin.txt"


def get_book():
    """Gutenberg #1228, from disk if setup already fetched it.

    `cut_prompts.py` runs during setup and caches the same book beside itself
    as `.gutenberg-1228.txt`, so on a machine this round built there is no
    reason to go to the network at all. On 2026-08-30 this function's single
    un-retried `urlopen` timed out on a healthy A100 whose engine had already
    started and warmed up, and took the configuration down with it -- 231 s of
    engine start thrown away for a text file that was already on the disk.
    """
    for p in (BOOK, os.path.join(D, ".gutenberg-1228.txt")):
        if os.path.exists(p) and os.path.getsize(p) > 400000:
            return open(p, encoding="utf-8", errors="ignore").read()
    urls = ("https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
            "https://www.gutenberg.org/files/1228/1228-0.txt")
    last = None
    for attempt in range(3):
        for url in urls:
            try:
                txt = urllib.request.urlopen(url, timeout=180).read().decode("utf-8", "ignore")
                if len(txt) > 400000:
                    open(BOOK, "w").write(txt)
                    return txt
                last = f"{url}: only {len(txt)} bytes"
            except Exception as e:
                last = f"{url}: {e!r}"
                log(f"get_book attempt {attempt + 1}: {last}")
        time.sleep(10)
    raise RuntimeError(f"could not fetch the book: {last}")


def ladder_for(model_dir):
    """one prompt per target, cut to that target in THIS model's tokens"""
    cache = f"{D}/ladder-{os.path.basename(model_dir)}.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    body = get_book()
    start = body.find("INTRODUCTION")
    body = body[start if start > 0 else 0:]
    ids = tok(body).input_ids
    out = []
    for t in TARGETS:
        take = ids[:t]
        text = tok.decode(take, skip_special_tokens=True)
        n = len(tok(text).input_ids)          # what it actually costs after decode
        out.append({"target": t, "prompt_tokens": n, "text": text})
    json.dump(out, open(cache, "w"))
    return out


def post(model, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.8, "stream": True,
        "stream_options": {"include_usage": True}, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, n, usage = None, 0, {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except Exception:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            ch = (ev.get("choices") or [{}])[0]
            if (ch.get("delta") or {}).get("content"):
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
    return ttft, n, time.time() - t0, usage


TOTAL_MIB = int(subprocess.run(
    "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits",
    shell=True, capture_output=True, text=True).stdout.strip().splitlines()[0])


def start_server(cfg, mml=None):
    """`mml` is the effective max-model-len for this attempt, not the constant.

    The Radeon runner has had a capacity retry since rev2; this one did not, and
    on 2026-08-30 that cost four L4 configurations. vLLM raises a ValueError when
    the KV pool cannot hold one request at `--max-model-len`, and the message
    carries the length that would fit. Without the retry the traceback that
    ValueError produces is caught by the crash test below and the configuration
    is recorded as a crash, which is what happened to B8, Q38S, G31 and Q38 --
    B8 by 0.13 GiB, needing 4.53 against 4.40 available.
    """
    mml = MML if mml is None else mml
    # `pkill -f 'vllm serve'` under shell=True can match its own shell, whose
    # command line contains the pattern. That is not theoretical: it left two
    # servers for one model alive at once today, and the second died in
    # init_device against a GPU the first still held -- recorded as a crash for
    # a configuration that had not been tried. The bracket stops the pattern
    # matching itself, and the wait confirms the port and the GPU are actually
    # free rather than assuming a sleep was long enough.
    # Killing the API server is not enough and waiting on a process list is not
    # enough either. vLLM's workers run as `VLLM::EngineCore`, whose command
    # line contains neither "vllm" nor "serve", so the parent dies and the
    # worker keeps the card: 72.7 GiB of 80 on this machine today, which made
    # the next configuration fail its own memory check and be recorded as a
    # crash it had nothing to do with. Kill both, then wait on the card itself.
    for pat in ("[v]llm serve", "[V]LLM::EngineCore", "vllm[.]model_executor"):
        subprocess.run(f"pkill -9 -f '{pat}' 2>/dev/null", shell=True)
    for _ in range(30):
        free = subprocess.run(
            "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits",
            shell=True, capture_output=True, text=True).stdout.strip()
        apps = subprocess.run(
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader",
            shell=True, capture_output=True, text=True).stdout.strip()
        # The A100 version waited for 70 GiB free, which never comes true on a
        # 23 GiB L4. Wait for most of whatever this card is instead.
        if not apps and free and int(free.splitlines()[0]) > 0.85 * TOTAL_MIB:
            break
        time.sleep(2)
    else:
        log("WARNING: card still held at start_server; the memory check may fail")
    time.sleep(2)
    # A stale log from a previous attempt reads as an instant crash -- the same
    # trap bench_runner.py's rev2 notes fixed on the Radeon side.
    lg0 = f"{D}/serve-{cfg['id']}.log"
    if os.path.exists(lg0):
        os.remove(lg0)
    mdir = f"{MODELS}/{cfg['model']}"
    # Prefix caching OFF, and this is the whole reason this round re-measures
    # the A100 rather than reusing it. Every rung of the ladder is a strict
    # prefix of the next -- ids truncated here, sentence boundaries on the
    # Radeon -- so with the cache on, a rung's prefill is charged only for the
    # tokens the previous rung did not already leave in the KV. On the A100
    # 2026-08-29 campaign (enable_prefix_caching=True) round 2 of the 32 K rung
    # took 0.201 s against round 1's 2.932 s, and a "prefill" of 159 299 tok/s
    # was recorded. The Radeon rows are clean -- its two rounds agree to 1.00x
    # at 32 K even on the arms whose config says True -- so the fix makes the
    # CUDA side match the ROCm side rather than the other way round.
    flags = [f"--max-model-len {mml}", f"--port {PORT}",
             "--no-enable-prefix-caching",
             f"--gpu-memory-utilization {cfg.get('util', 0.90)}"]
    if cfg.get("dtype"):
        flags.append(f"--dtype {cfg['dtype']}")
    if cfg.get("mns"):
        flags.append(f"--max-num-seqs {cfg['mns']}")
    if cfg["model"].startswith("gemma-4"):
        flags.append(GEMMA_MM)
    if cfg.get("spec"):
        flags.append("--speculative-config '" + json.dumps(cfg["spec"]) + "'")
    sc = f"{D}/serve-{cfg['id']}.sh"
    with open(sc, "w") as fh:
        fh.write("#!/bin/bash\nset -u\n")
        fh.write(f"exec vllm serve {mdir} " + " ".join(flags) +
                 f" > {D}/serve-{cfg['id']}.log 2>&1\n")
    os.chmod(sc, 0o755)
    subprocess.Popen(["bash", sc])
    t0, hard, stall = time.time(), 3600, 600
    lg = f"{D}/serve-{cfg['id']}.log"
    last = 0
    while time.time() - t0 < hard:
        txt = open(lg).read() if os.path.exists(lg) else ""
        if "Application startup complete" in txt:
            return "ready", txt
        # Before the crash test: this condition raises a ValueError, so its own
        # traceback would otherwise be read as a crash. The message names the
        # length that would fit; 0.27 has a second phrasing with no number.
        m = re.search(r"estimated maximum model length is (\d+)", txt)
        if m:
            return "capacity", int(m.group(1))
        if "No available memory for the cache blocks" in txt:
            return "capacity", -1
        # torch logs whole formatted tracebacks at W level: triton_bundler
        # prints one per missing AOT cubin when it falls back to recompiling,
        # and injecting #45450 mid-run invalidates exactly that cache. The
        # naive test stopped a healthy server on the Radeon side today. A real
        # traceback sits at the head of its line behind only the process tag;
        # a logged one carries its logger's "<file>.py:<line>]" ahead of it.
        real_tb = [l for l in txt.splitlines()
                   if "Traceback (most recent call last)" in l
                   and not re.search(r"\.py:\d+\]", l.split("Traceback")[0])]
        if real_tb or "EngineCore failed to start" in txt \
                or "Engine core initialization failed" in txt:
            return "crash", txt[-2500:]
        idle = time.time() - os.path.getmtime(lg) if os.path.exists(lg) else time.time() - t0
        if idle > stall:
            return "timeout", f"log idle {idle:.0f}s"
        el = time.time() - t0
        if el - last > 240:
            last = el
            log(f"{cfg['id']}: still starting ({el/60:.0f} min)")
        time.sleep(5)
    return "timeout", "hard cap"


def meta_from(cfg_id, txt):
    emit(describe())
    m = {"kind": "model_meta", "cfg": cfg_id, "machine": MACHINE,
         "vram_total_mib": TOTAL_MIB}
    for k, p in {"init_engine_s": r"init engine[^\n]*took ([0-9.]+) s",
                 "model_load_s": r"Model loading took [0-9.]+ GiB(?: memory)? and ([0-9.]+) seconds",
                 "kv_gib": r"Available KV cache memory: ([0-9.]+) GiB",
                 "kv_tokens": r"GPU KV cache size: ([\d,]+) tokens",
                 # 0.28 writes this two ways from two branches of cuda.py:
                 # "Using AttentionBackendEnum.TRITON_ATTN backend." and
                 # "Using FLASH_ATTN attention backend out of potential ...".
                 # A regex for one silently misses the other, which is why the
                 # A100 campaign recorded no backend at all.
                 "backend": r"Using (?:AttentionBackendEnum\.)?([A-Z0-9_]+)(?: attention)? backend",
                 "wna16_kernel": r"Using (\w+) for CompressedTensorsWNA16",
                 "prefix_caching": r"enable_prefix_caching=(\w+)"}.items():
        mm = re.search(p, txt)
        if mm:
            m[k] = mm.group(1).replace(",", "")
    return m


def inject_45450():
    """Apply the mechanism the spec article validated, once, before it is needed.

    Not the PR's diff: that no longer applies to any tree here. This is
    benchmarks/cuda-a100/45450-validation/inject_45450.py, whose anchors are
    literal source lines and whose assertions fail loudly rather than half-patch.
    """
    import importlib
    r = subprocess.run([sys.executable, f"{D}/inject_45450.py"],
                       capture_output=True, text=True)
    log("inject_45450: " + (r.stdout + r.stderr).strip()[-300:])
    return r.returncode == 0


def run_cfg(cfg, done):
    cid = cfg["id"]
    if ("cfg", cid) in done:
        log(f"{cid}: already complete, skip")
        return
    if cfg.get("p45450") and not globals().get("_INJECTED"):
        if not inject_45450():
            emit({"kind": "config_failed", "cfg": cid, "why": "inject_45450 failed"})
            return
        globals()["_INJECTED"] = True
    mml = MML
    info = None
    for _ in range(4):
        st, info = start_server(cfg, mml)
        if st == "ready":
            break
        if st == "capacity":
            if info == -1:
                mml = max(1200, mml // 2)
                log(f"{cid}: no room for KV -> retry mml {mml}")
                emit({"kind": "note", "cfg": cid, "note": f"no-kv-room, mml->{mml}"})
                continue
            if info < 2000:
                log(f"{cid}: KV holds only {info} tok -> not measurable, FAILED")
                emit({"kind": "config_failed", "cfg": cid,
                      "why": f"kv_max_len={info} too small at util={cfg.get('util', 0.90)}"})
                return
            newmml = max(1200, int(info * 0.99))
            log(f"{cid}: KV holds only {info} tok -> retry mml {newmml}")
            emit({"kind": "note", "cfg": cid, "note": f"kv_max_len={info}, mml->{newmml}"})
            mml = newmml
            continue
        log(f"{cid}: {st}, FAILED")
        emit({"kind": "config_failed", "cfg": cid, "why": st, "tail": str(info)[-1200:]})
        return
    else:
        emit({"kind": "config_failed", "cfg": cid, "why": "startup retries exhausted"})
        return
    emit(meta_from(cid, info) | {"mml": mml, "util": cfg.get("util", 0.90)})
    # One discarded request before the ladder. Without it the very first
    # measurement of the run -- prefill, round 1, the 500 rung -- absorbs
    # everything a cold engine does once: the first CUDA graph replay, the
    # first allocation out of the KV pool, lazy JIT. On the L4 that made the
    # 500 rung 2.064 s against its own round 2's 0.287 s, a 151 % spread on a
    # rung whose every other round agrees to 0.07 %, and cost the rung its
    # chart grade. The Radeon runner has always had this, as its health gate;
    # a100_run.py never did, so every CUDA config in this repository has one
    # ungraded rung for a reason that is the harness and not the machine.
    try:
        post(f"{MODELS}/{cfg['model']}", "Say OK briefly.", 8, 180)
        log(f"{cid}: warmup ok")
    except Exception as ex:
        log(f"{cid}: warmup failed {ex!r} (continuing)")
        emit({"kind": "note", "cfg": cid, "note": f"warmup failed: {ex!r}"[:200]})
    lad = ladder_for(f"{MODELS}/{cfg['model']}")
    ok = err = 0
    for e in lad:
        if e["prompt_tokens"] + GEN + 100 > mml:
            log(f"{cid}: target {e['target']} exceeds mml, stop")
            break
        for rnd in range(1, 6 if e["target"] == 500 else 3):
            if (cid, e["target"], rnd) in done:
                ok += 1
                continue
            try:
                smp = Sampler()
                with smp:
                    ttft, n, wall, usage = post(f"{MODELS}/{cfg['model']}",
                                                e["text"], GEN, 900)
                dec = (n - 1) / (wall - ttft) if ttft and wall > ttft and n > 1 else 0.0
                tele = dict(smp.result, wall_s=round(wall, 3))
                emit({"kind": "prefill", "cfg": cid, "machine": MACHINE,
                      "target": e["target"], "round": rnd,
                      "prompt_tokens": usage.get("prompt_tokens", e["prompt_tokens"]),
                      "ttft": round(ttft or 0, 4), "gen_tokens": 0,
                      "prefill_tps": round((usage.get("prompt_tokens") or e["prompt_tokens"]) / ttft, 1)
                      if ttft else 0} | tele)
                emit({"kind": "decode", "cfg": cid, "machine": MACHINE,
                      "target": e["target"], "round": rnd,
                      "prompt_tokens": usage.get("prompt_tokens", e["prompt_tokens"]),
                      "gen_tokens": n, "decode_tps": round(dec, 4)} | tele)
                ok += 1
                log(f"{cid}: {e['target']} r{rnd} {dec:.2f} tok/s")
            except Exception as ex:
                err += 1
                log(f"{cid}: {e['target']} r{rnd} ERROR {ex!r}")
                emit({"kind": "error", "cfg": cid, "target": e["target"], "round": rnd,
                      "err": repr(ex)[:400]})
                if err >= 4:
                    emit({"kind": "config_failed", "cfg": cid, "why": "too many errors"})
                    return
    emit({"kind": "config_complete", "cfg": cid, "ok": ok, "err": err})
    log(f"{cid}: COMPLETE ({ok} ok, {err} err)")
    # A Colab VM can be reclaimed without warning; two were, at 15:58 and 16:56.
    # A copy inside the VM protects against nothing -- both copies go with it.
    # The results are printed here instead, so they reach the caller's terminal
    # and survive the machine that produced them. The poller on the other end
    # writes them down.
    try:
        print("=== HARVEST BEGIN " + cid + " ===", flush=True)
        for line in open(RES):
            print("H|" + line.rstrip(), flush=True)
        print("=== HARVEST END " + cid + " ===", flush=True)
    except Exception as ex:
        log(f"harvest failed: {ex!r}")


if __name__ == "__main__":
    os.makedirs(D, exist_ok=True)
    log(f"sys.argv as seen by the runner: {sys.argv!r}")
    want = None   # ipykernel's argv is not ours; the config list is the filter
    import vllm
    # Record the stack. Nothing in the 2026-08-29 A100 logs names a torch or a
    # CUDA version, and the L4's were lost with its VM, so both had to be left
    # null in the projection. A run that does not write down what it ran on
    # cannot be compared to one that did.
    def _v(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None
    import torch
    smi = subprocess.run(
        "nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version "
        "--format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()
    emit({"kind": "run_meta", "machine": MACHINE, "vllm": vllm.__version__,
          "torch": _v("torch"), "transformers": _v("transformers"),
          "cuda": torch.version.cuda, "gpu": smi})
    log(f"=== {MACHINE} run start, vllm {vllm.__version__}, torch {_v('torch')}, "
        f"cuda {torch.version.cuda} ===")
    done = done_keys()
    for cfg in CFGS:
        if want and cfg["id"] not in want:
            continue
        try:
            run_cfg(cfg, done)
        except Exception as ex:
            log(f"{cfg['id']}: unhandled {ex!r}")
            emit({"kind": "config_failed", "cfg": cfg["id"], "why": repr(ex)[:400]})
    subprocess.run("pkill -f 'vllm serve' 2>/dev/null", shell=True)
    log(f"=== {MACHINE} run end ===")
    print("A100_CAMPAIGN_DONE", flush=True)
