# 对外
H I
## 目标
实现显示屏链接单片机，控制面板的内置
### 原理
#### 1.有关于显示屏链接单片机的散热
对于内置于机器人的显示屏，其机器人内部是不透气的，也就是说，单片机运转的余机械热量会留存于机器人内部环境，可能会导致包括但不限于 机器热损毁 机器人爆炸 线路烧毁 机器人异常肿大等问题 其散热能力之差，比热容容忍度之小，故采用水冷法支持散热问题，通过液体之比热容大的特点，借用液体来进行散热工作https://kns.cnki.net/nzkhtml/xmlRead/trialRead.html?dbCode=CJFD&tableName=CJFDTOTAL&fileName=CFXB202406012&fileSourceType=1&invoice=bYRDtixbFHVQQ4YuPg3qBwCtDBugoJEp8eyw%2fr%2bOWN8k4f2XOW7liGw9NlqhXLHdSHrDJxDZeztA5RgxHmHUTRGq8Z2xBq1a%2bP%2fVRc0tsmY1Eg13VxuOqNSQge0UAHWA4XmLYy2rff13W5D4I5bS6KaQgjCYKvRI3gDTRhRtpoA%3d&appId=KNS_BASIC_PSMC
***
#### 2.显示屏链接单片机的实现
对于显示屏来说，单个显示屏需要连接单片机完成编程和功能实现，也就是说，需要在单片机里面编程，然后将其输出至显示屏中，也就是说需要在无系统的单片机上进行编程，并且要适用
https://kns.cnki.net/kcms2/article/abstract?v=_W1AupcyYgb5rwDSDnhBzXYGNavek3s6dv9KAficJgXp00RSqb6ExxCcVTTiR9qa-y0v1JDUCBl-qQiDyun7pK7hZKi7LSe1WwZ2cBIU39cuTCMQc5TXO-O3ZcSnEgApPhBbUNptfWTL4kKv70evy1wi7NQ0pafjRFUaiunOsVodURPs-CvSReQ_OTl_DHQ-&uniplatform=NZKPT&language=CHS

以及

https://blog.csdn.net/cqtianxingkeji/article/details/134655385
***
#### 3.有关于控制面板的设计
为了照顾老年人无法完美使用智能手机的问题，这边选择使用触摸屏来代替智能手机操作，智能触摸屏比智能手机的优点有以下几项 1.老年人操作方便 2.图像提示直观，入手更快 3.操作屏简化了操作
（以下论文不用完全抄上来，重点放如何去实现的部分）

https://d.wanfangdata.com.cn/thesis/D243796
***
#### 4.有关于控制面板的操控
对于第四点我们需要实现：
搜索
拍照
预警
定位
AR导航
music
虚拟形象

于是，我们写了一个集成式软件，用于实现以上功能
实现方式如下：
对于搜索 我们选择直接调用百度的搜索引擎

对于拍照 为了检测伤口等信息，拍摄功能是必须的 我们采用ai照相机的方式来实现这一功能 https://blog.csdn.net/gitblog_00025/article/details/139792008

对于预警 我们的设定是，当ai发现老人的伤口较为严重，无法依靠单人/非医疗机构处理时，便会接上报警和 https://gitee.com/liuyueyi/quick-alarm

对于定位 本质上来说，机器人本身就是小型网络架构，可以通过wifi来进行精准定位，此功能旨在防止老人前往河边，湖边等危险，安全没有保障等地方
https://blog.csdn.net/gitblog_00075/article/details/139913874

对于AR导航 由于导航和空间认知能力下降，复杂的室内，外环境中的寻路对于老年人来说通常具有挑战性。因此采用AR导航来保障老年人在行走/出行途中避免摔倒/遭遇意外
https://www.x-mol.com/paper/1727560013532909568/t


对于音乐 我们直接内置了大量藏语/秦腔/京剧/越剧经典歌曲，可以直接播放
对于虚拟形象 我们采用《超能陆战队》中的主角 大白 作为基础模板，再次上面改造，形成了我们的虚拟形象
