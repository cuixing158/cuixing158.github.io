hexo clean
hexo g
hexo s
hexo d

记得在此目录下git bash here执行以上命令，即可完成发布


只需要在source\_posts文件下写博客md文件和图像文件即可?
tips: 段落首行缩进方法，在开头的时候，先输入这个，然后紧跟着输入文本即可。分号也不要掉: &#160; &#160; &#160; &#160;



github远程端有3个分支，一个用于展示的个人网页，一个用于写博客的材料文件，另一个为博客模板
博客材料分支提交：

git add .
git commit -m"说明"
git push origin my_blog
