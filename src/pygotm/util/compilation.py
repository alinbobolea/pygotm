"""
module gotm_compilation
   character(len=*),parameter :: compiler = ""
   character(len=*),parameter :: compiler_id = ""
   character(len=*),parameter :: compiler_version = ""
end module
"""

__all__ = ["compiler", "compiler_id", "compiler_version"]

compiler: str = ""
compiler_id: str = ""
compiler_version: str = ""
