from collections import namedtuple

class InstallCommandError(Exception):
    """Thrown when the installation of a third-party package
    fails

    Arguments
    ---------
    message : str
        The error message
    code : int, optional
        The return code of the command
    out : str
        The standard output of the command
    err : str
        The captured standard error stream of the command
    """
    def __init__(self, message, code=None, out=None, err=None) -> None:
        detailed_message = message
        if code is not None:
            message += f"\nReturn Code: {code}"
        if out:
            message += f"\nOutput: {out}"
        if err:
            message += f"\Error: {err}"
        super().__init__(detailed_message)
        self.code = code
        self.out = out
        self.err = err


TTSInferenceResult = namedtuple(
    "TTSInferenceResult",
    ["wav", "length", "tokens"]
)