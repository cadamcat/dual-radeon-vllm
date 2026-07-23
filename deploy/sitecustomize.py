# Pre-bind amdsmi's real rocm_smi/rsmi BEFORE torch loads the no-hostcall
# librccl, whose rsmi stub would otherwise shadow amdsmi's rsmi symbols and
# break device enumeration (AMDSMI_STATUS_NOT_INIT). Keeping amdsmi initialized
# (no shut_down) holds the native libs loaded so rsmi stays bound to the real
# rocm_smi for the life of the process.
try:
    import amdsmi
    amdsmi.amdsmi_init()
except Exception:
    pass
