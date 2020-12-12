import visdom
from PIL import Image, ImageDraw
from torchnet.logger import VisdomPlotLogger
import torch
import datetime
import numpy as np
from utils import defappendTensorToTensor


def createDrawAvatar(xSize, ySize):
    avatar = Image.new("RGB", [xSize, ySize], (255, 255, 255))
    drawAvatar = ImageDraw.Draw(avatar)
    return avatar, drawAvatar


class classDrawHeatMap(object):
    def __init__(self, options, viz, edgeSet, drawAvatar, avatar):
        self.viz = viz
        self.edgeSet = edgeSet
        self.drawAvatar = drawAvatar
        self.avatar = avatar
        self.options = options

    def draw(self, flowData):
        self.viz.text('Time: {} | | Update heatmap'.format(datetime.datetime.now()), win='Notices:',
                      opts={'title': 'Notices'})
        flowData = flowData[:, :, 0]
        flowData = (flowData - flowData.mean()) / flowData.std()
        for layerCT, flow in enumerate(flowData.permute(-1, 0, 1, 2)):
            flowData = flow.reshape(21, 4, 3).mean(-1)
            for edgeCount, edge in enumerate(self.edgeSet):
                nodeFlow = flowData[edgeCount]
                for dirCount, dir in enumerate(edge):
                    if nodeFlow[dirCount] >= 0.75:
                        self.drawAvatar.line(dir.tolist(), (255, 0, 0), width=3)
                    elif 0.5 <= nodeFlow[dirCount] < 0.75:
                        self.drawAvatar.line(dir.tolist(), (255, 102, 102), width=3)
                    elif 0.25 <= nodeFlow[dirCount] < 0.5:
                        self.drawAvatar.line(dir.tolist(), (255, 153, 153), width=3)
                    elif nodeFlow[dirCount] < 0.25:
                        self.drawAvatar.line(dir.tolist(), (0, 0, 0), width=3)
                    else:
                        ValueError('Wrong heatmap input: {}'.format(edgeCount, dirCount, flowData, nodeFlow[dirCount]))
                    self.viz.image(np.array(self.avatar).transpose([2, 0, 1]), win='Layer-{}'.format(layerCT),
                                   opts={'title': 'Layer-{}'.format(layerCT)})

class Global_Speed_Logger(object):
    def __init__(self, options):
        self.avg_speed_sum_avg = None
        self.avg_speed_sum = 0.
        self.avg_iql = VisdomPlotLogger(
            'line',
            port=8100,
            win='avg_speed',
            env=options.visdom_env,
            opts={
                'title': 'Global Avg. Speed',
                'xlabel': 'Time',
                'ylabel': 'm/s'
            })
        self.options = options
        self.globalSpeedPool =None
        self.globalVelPool = None

    def logSpeed(self, global_CT, speed):
        self.globalSpeedPool = defappendTensorToTensor(self.globalSpeedPool, speed, new_axis=0)
        globalTmpSpeed = self.globalSpeedPool.mean()
        self.avg_iql.log(global_CT, speed, name='IAS-{}'.format(self.options.projectname))
        self.avg_iql.log(global_CT, globalTmpSpeed, name='GAS-{}'.format(self.options.projectname))
        self.avg_iql.log(global_CT, self.globalSpeedPool.std(), name='std-{}'.format(self.options.projectname))

    def log(self, global_CT, speed, velNum):

        self.globalSpeedPool = defappendTensorToTensor(self.globalSpeedPool, speed, new_axis=0)
        self.globalVelPool = defappendTensorToTensor(self.globalVelPool, velNum, new_axis=0)
        tmpSpeed = speed[velNum!=0].mean()
        globalTmpSpeed = self.globalSpeedPool[self.globalVelPool!=0].mean()
        self.avg_iql.log(global_CT, tmpSpeed, name='IAS-{}'.format(self.options.projectname))
        self.avg_iql.log(global_CT, globalTmpSpeed, name='GAS-{}'.format(self.options.projectname))

class Global_PassNum_Logger(object):
    def __init__(self, options):
        self.avg_sum_avg = None
        self.avg_speed_sum = 0.
        self.avg_Pass_num = VisdomPlotLogger(
            'line',
            port=8100,
            win='Average Halting Num',
            env=options.visdom_env,
            opts={
                'title': 'Global Halting Num',
                'xlabel': 'Period',
                'ylabel': 'Car'
            })
        self.options = options

    def log(self, period_CT, passnum):
        self.avg_Pass_num.log(period_CT, passnum, name='IAH-{}'.format(self.options.projectname))

class EdgeInfo_Logger(object):
    def __init__(self, options):
        self.options = options
        self.travelTimelogger = VisdomPlotLogger(
            'line',
            port=8100,
            win='Average Travel Time',
            env=options.visdom_env,
            opts={
                'title': 'Global Avg. Travel Time',
                'xlabel': 'Period',
                'ylabel': 's'
            })
        self.traveltimePool = 0.
        self.waittimePool = 0.
        self.waittimelogger = VisdomPlotLogger(
            'line',
            port=8100,
            win='Average waiting Time',
            env=options.visdom_env,
            opts={
                'title': 'Global Avg. waiting Time',
                'xlabel': 'Period',
                'ylabel': 's'
            })

    def logtravelTime(self, period_CT, time):
        self.traveltimePool+=time
        if period_CT==0:
            averagetime = time
        else:
            averagetime = self.traveltimePool / (period_CT+1)
        self.travelTimelogger.log(period_CT, averagetime, name='GAH-{}'.format(self.options.projectname))
        self.travelTimelogger.log(period_CT, time, name='IAH-{}'.format(self.options.projectname))

    def logwaitTime(self, period_CT, time):
        self.waittimePool+=time
        if period_CT==0:
            averagetime = time
        else:
            averagetime = self.waittimePool / (period_CT+1)
        self.waittimelogger.log(period_CT, averagetime, name='GAH-{}'.format(self.options.projectname))
        self.waittimelogger.log(period_CT, time, name='IAH-{}'.format(self.options.projectname))


class GlobalActionLossLogger(object):
    def __init__(self, options):
        self.LossLogger = VisdomPlotLogger(
            'line',
            port=8100,
            win='Actor Loss',
            env=options.visdom_env,
            opts={
                'title': 'Actor Loss',
                'xlabel': 'Period',
                'ylabel': 'Loss'
            })
        self.options = options

    def log(self, period_CT, loss, valueloss):
        self.LossLogger.log(period_CT, loss, name='actorloss-{}'.format(self.options.projectname))
        self.LossLogger.log(period_CT, valueloss, name='valueloss-{}'.format(self.options.projectname))

class RewardLogger(object):
    def __init__(self, options):
        self.LossLogger = VisdomPlotLogger(
            'line',
            port=8100,
            win='Reward vs. Value',
            env=options.visdom_env,
            opts={
                'title': 'Reward vs. Value',
                'xlabel': 'Period',
            })
        self.options = options

    def log(self, period_CT, reward):
        self.LossLogger.log(period_CT, reward, name='reward-{}'.format(self.options.projectname))
        #self.LossLogger.log(period_CT, value, name='value-{}'.format(self.options.projectname))


def defGetEdgeSet(ySize):
    edgeSet = torch.zeros(21, 4, 4)
    X = torch.zeros(10)
    Y = torch.zeros(10)
    for i in range(10):
        if i % 2 == 0:
            Y[i] = ySize - 50 - 58 * (i / 2)
        else:
            Y[i] = ySize - 58 - 58 * (i / 2)
    for i in range(10):
        if i % 2 == 0:
            X[i] = 50 + 58 * (i / 2)
        else:
            X[i] = 58 + 58 * (i / 2)

    edgeSet[0, 0] = torch.Tensor([X[1], Y[3], X[1], Y[4]])
    edgeSet[0, 1] = torch.Tensor([X[1], Y[2], X[2], Y[2]])
    edgeSet[0, 2] = torch.Tensor([X[0], Y[2], X[2], Y[0]])

    edgeSet[1, 0] = torch.Tensor([X[1], Y[5], X[1], Y[6]])
    edgeSet[1, 1] = torch.Tensor([X[1], Y[4], X[2], Y[4]])
    edgeSet[1, 2] = torch.Tensor([X[0], Y[4], X[0], Y[3]])

    edgeSet[2, 0] = torch.Tensor([X[1], Y[7], X[2], Y[8]])
    edgeSet[2, 1] = torch.Tensor([X[1], Y[6], X[2], Y[6]])
    edgeSet[2, 2] = torch.Tensor([X[0], Y[6], X[0], Y[5]])

    edgeSet[3, 0] = torch.Tensor([X[3], Y[1], X[3], Y[2]])
    edgeSet[3, 1] = torch.Tensor([X[3], Y[0], X[4], Y[0]])
    edgeSet[3, 3] = torch.Tensor([X[2], Y[1], X[1], Y[2]])

    edgeSet[4, 0] = torch.Tensor([X[3], Y[3], X[3], Y[4]])
    edgeSet[4, 1] = torch.Tensor([X[3], Y[2], X[4], Y[2]])
    edgeSet[4, 2] = torch.Tensor([X[2], Y[2], X[2], Y[1]])
    edgeSet[4, 3] = torch.Tensor([X[2], Y[3], X[1], Y[3]])
    edgeSet[5, 0] = torch.Tensor([X[3], Y[5], X[3], Y[6]])
    edgeSet[5, 1] = torch.Tensor([X[3], Y[4], X[4], Y[4]])
    edgeSet[5, 2] = torch.Tensor([X[2], Y[4], X[2], Y[3]])
    edgeSet[5, 3] = torch.Tensor([X[2], Y[5], X[1], Y[5]])
    edgeSet[6, 0] = torch.Tensor([X[3], Y[7], X[3], Y[8]])
    edgeSet[6, 1] = torch.Tensor([X[3], Y[6], X[4], Y[6]])
    edgeSet[6, 2] = torch.Tensor([X[2], Y[6], X[2], Y[5]])
    edgeSet[6, 3] = torch.Tensor([X[2], Y[7], X[1], Y[7]])

    edgeSet[7, 1] = torch.Tensor([X[3], Y[8], X[4], Y[8]])
    edgeSet[7, 2] = torch.Tensor([X[2], Y[8], X[2], Y[7]])
    edgeSet[7, 3] = torch.Tensor([X[2], Y[9], X[0], Y[7]])

    edgeSet[8, 0] = torch.Tensor([X[5], Y[1], X[5], Y[2]])
    edgeSet[8, 1] = torch.Tensor([X[5], Y[0], X[6], Y[0]])
    edgeSet[8, 3] = torch.Tensor([X[4], Y[1], X[3], Y[1]])

    edgeSet[9, 0] = torch.Tensor([X[5], Y[3], X[5], Y[4]])
    edgeSet[9, 1] = torch.Tensor([X[5], Y[2], X[6], Y[2]])
    edgeSet[9, 2] = torch.Tensor([X[4], Y[2], X[4], Y[1]])
    edgeSet[9, 3] = torch.Tensor([X[4], Y[3], X[3], Y[3]])
    edgeSet[10, 0] = torch.Tensor([X[5], Y[5], X[5], Y[6]])
    edgeSet[10, 1] = torch.Tensor([X[5], Y[4], X[6], Y[4]])
    edgeSet[10, 2] = torch.Tensor([X[4], Y[4], X[4], Y[3]])
    edgeSet[10, 3] = torch.Tensor([X[4], Y[5], X[3], Y[5]])
    edgeSet[11, 0] = torch.Tensor([X[5], Y[7], X[5], Y[8]])
    edgeSet[11, 1] = torch.Tensor([X[5], Y[6], X[6], Y[6]])
    edgeSet[11, 2] = torch.Tensor([X[4], Y[6], X[4], Y[5]])
    edgeSet[11, 3] = torch.Tensor([X[4], Y[7], X[3], Y[7]])

    edgeSet[12, 1] = torch.Tensor([X[5], Y[8], X[6], Y[8]])
    edgeSet[12, 2] = torch.Tensor([X[4], Y[8], X[4], Y[7]])
    edgeSet[12, 3] = torch.Tensor([X[4], Y[9], X[3], Y[9]])

    edgeSet[13, 0] = torch.Tensor([X[7], Y[1], X[7], Y[2]])
    edgeSet[13, 1] = torch.Tensor([X[7], Y[0], X[9], Y[2]])
    edgeSet[13, 3] = torch.Tensor([X[6], Y[1], X[5], Y[1]])

    edgeSet[14, 0] = torch.Tensor([X[7], Y[3], X[7], Y[4]])
    edgeSet[14, 1] = torch.Tensor([X[7], Y[2], X[8], Y[2]])
    edgeSet[14, 2] = torch.Tensor([X[6], Y[2], X[6], Y[1]])
    edgeSet[14, 3] = torch.Tensor([X[6], Y[3], X[5], Y[3]])
    edgeSet[15, 0] = torch.Tensor([X[7], Y[5], X[7], Y[6]])
    edgeSet[15, 1] = torch.Tensor([X[7], Y[4], X[8], Y[4]])
    edgeSet[15, 2] = torch.Tensor([X[6], Y[4], X[6], Y[3]])
    edgeSet[15, 3] = torch.Tensor([X[6], Y[5], X[5], Y[5]])
    edgeSet[16, 0] = torch.Tensor([X[7], Y[7], X[7], Y[8]])
    edgeSet[16, 1] = torch.Tensor([X[7], Y[6], X[8], Y[6]])
    edgeSet[16, 2] = torch.Tensor([X[6], Y[6], X[6], Y[5]])
    edgeSet[16, 3] = torch.Tensor([X[6], Y[7], X[5], Y[7]])

    edgeSet[17, 1] = torch.Tensor([X[7], Y[8], X[8], Y[7]])
    edgeSet[17, 2] = torch.Tensor([X[6], Y[8], X[6], Y[7]])
    edgeSet[17, 3] = torch.Tensor([X[6], Y[9], X[5], Y[9]])

    edgeSet[18, 0] = torch.Tensor([X[9], Y[3], X[9], Y[4]])
    edgeSet[18, 2] = torch.Tensor([X[8], Y[2], X[7], Y[1]])
    edgeSet[18, 3] = torch.Tensor([X[8], Y[3], X[7], Y[3]])

    edgeSet[19, 0] = torch.Tensor([X[9], Y[5], X[9], Y[6]])
    edgeSet[19, 2] = torch.Tensor([X[8], Y[4], X[8], Y[3]])
    edgeSet[19, 3] = torch.Tensor([X[8], Y[5], X[7], Y[5]])

    edgeSet[20, 0] = torch.Tensor([X[9], Y[7], X[7], Y[9]])
    edgeSet[20, 2] = torch.Tensor([X[8], Y[6], X[8], Y[5]])
    edgeSet[20, 3] = torch.Tensor([X[8], Y[7], X[7], Y[7]])
    return edgeSet





class visTools(object):
    def __init__(self, options):
        ySize = 340
        xSize = 340
        self.viz = visdom.Visdom(port=8100, env='drawHeatMap')
        self.avatar, self.drawAvatar = createDrawAvatar(xSize,ySize)
        self.loss_plot = GlobalActionLossLogger(options)

        self.edgeSet = defGetEdgeSet(ySize)

        self.global_speed_logger = Global_Speed_Logger(options)
        self.global_PassNum_Logger =Global_PassNum_Logger(options)
        self.global_traveltime_logger=EdgeInfo_Logger(options)
        self.rewardLogger = RewardLogger(options)

    def defDrawHeatMap(self, flowData):
        self.viz.text('Time: {} | | Update heatmap'.format(datetime.datetime.now()), win='Notices:',
                 opts={'title': 'Notices'})
        flowData = flowData[:, :, 0]
        flowData = (flowData - flowData.mean()) / flowData.std()
        for layerCT, flow in enumerate(flowData.permute(-1, 0, 1, 2)):
            flowData = flow.reshape(21, 4, 3).mean(-1)
            for edgeCount, edge in enumerate(self.edgeSet):
                nodeFlow = flowData[edgeCount]
                for dirCount, dir in enumerate(edge):
                    if nodeFlow[dirCount] >= 0.75:
                        self.drawAvatar.line(dir.tolist(), (255, 0, 0), width=3)
                    elif 0.5 <= nodeFlow[dirCount] < 0.75:
                        self.drawAvatar.line(dir.tolist(), (255, 102, 102), width=3)
                    elif 0.25 <= nodeFlow[dirCount] < 0.5:
                        self.drawAvatar.line(dir.tolist(), (255, 153, 153), width=3)
                    elif nodeFlow[dirCount] < 0.25:
                        self.drawAvatar.line(dir.tolist(), (0, 0, 0), width=3)
                    else:
                        ValueError('Wrong heatmap input: {}'.format(edgeCount, dirCount, flowData, nodeFlow[dirCount]))
                    self.viz.image(np.array(self.avatar).transpose([2, 0, 1]), win='Layer-{}'.format(layerCT),
                              opts={'title': 'Layer-{}'.format(layerCT)})