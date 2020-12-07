# IVMM
Implement of IVMM map matching method.

## Getting Start
**Input file (.csv) sample**:<br />

| plate | color | longitude | latitude | timestamp | velocity | status |
| ----- |:-----:| ---------:| --------:| ---------:| --------:| ------:|
|粤123456|蓝的|113.961098|22.553101|2014-07-03 00:00:02|17|0|
|粤123456|蓝的|113.962303|22.547001|2014-07-03 00:01:48|21|0|
|粤123456|蓝的|113.962997|22.547001|2014-07-03 00:02:18|0|0|

**Output file (.csv) sample**:

| |i_p_i|e_i|end_node|edge_progress|x|y|oneway|length|u|v|plate|longitude|latitude|timestamp|velocity|dis_f_pre|
| ----- |:-----:| ---------:| --------:| ---------:| --------:| ------:| ------:| ------:| ------:| ------:| ------:| ------:| ------:| ------:| ------:| ------:|
|0|877675.0|45266|0|0.7777777777777778|113.961161|22.55314|True|205.163|1116415224|1116415555|粤B0AA56|113.961098|22.553101|2014-07-03 00:00:02|17||
|1|1419978.0|72794|0|0.07142857142857142|113.962331|22.547002|False|140.21699999999998|2528898679|2528898707|粤B0AA56|113.962303|22.547001|2014-07-03 00:01:48|21|689.4855647331082|
|2|1419997.0|72795|0|0.4166666666666667|113.962945|22.54705|False|127.939|2528898679|2528898834|粤B0AA56|113.962997|22.547001|2014-07-03 00:02:18|0|71.27096668179584|

Please refer to the paper for details of the method: @inproceedings{yuan2010interactive,
  title={An interactive-voting based map matching algorithm},
  author={Yuan, Jing and Zheng, Yu and Zhang, Chengyang and Xie, Xing and Sun, Guang-Zhong},
  booktitle={2010 Eleventh international conference on mobile data management},
  pages={43--52},
  year={2010},
  organization={IEEE}
}

Issues are welcome.
