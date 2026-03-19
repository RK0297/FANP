@echo off
cd C:\Users\Radhakrishna\Desktop\DL_PROJECT
git status
echo.
echo Committing changes...
git add .gitignore
git commit -m "chore: update gitignore to track source code"
git log --oneline -1
echo.
echo Pushing to GitHub...
git push
