"""
module gotm_version
   character(len=*),parameter :: git_commit_id = "4.1.0"
   character(len=*),parameter :: git_branch_name = "master"
end module
"""

__all__ = ["git_commit_id", "git_branch_name"]

git_commit_id: str = "4.1.0"
git_branch_name: str = "master"
