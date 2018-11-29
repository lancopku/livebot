'''
 @Date  : 8/14/2018
 @Author: Shuming Ma
 @mail  : shumingma@pku.edu.cn 
 @homepage: shumingma.com
'''
import os

def extract_comments(filename, outfile, video_id):
	with open(outfile, 'w', encoding='utf8') as fw:
		for line in open(filename, 'r', encoding='utf8').readlines():
			if line.startswith('Dialogue:'):
				comment = line[line.rfind('}')+1:].strip()
				time = line[line.find(',')+1:line.find('.')]
				time = sum(x * int(t) for x, t in zip([3600, 60, 1], time.split(":")))
				fw.write("%d\t%d\t%s\n" % (video_id, time, comment))


def extract_frame(filename, outdir):
	if not os.path.exists(outdir):
		os.mkdir(outdir)

	cmd_str = 'ffmpeg -i "%s" -r 1/1 -s 224x224 -f image2 %s/' % (filename, outdir) + '%d.bmp'
	print(cmd_str)
	os.system(cmd_str)


if __name__ == '__main__':

	count = 0

	for dir, _, filenames in os.walk('video/'):
		for filename in filenames:
			if filename.endswith('flv'):
				title = filename[:-4]
				if os.path.exists(os.path.join(dir, title+".ass")):
					extract_frame(os.path.join(dir, filename), 'img/%d' % count)
					extract_comments(os.path.join(dir, title+".ass"), 'comment/%d.txt' % count, count)
					count += 1

	print(count)