# ICan
Begin now!


House Prices: Advanced Regression Techniques。

# 认识数据：
House Prices数据集分为train（即训练）数据和test（即测试）数据，
其中，训练集含有1460个样本，80个属性（包括序号），一个标签（SalePrice，即房价）；测试集含有1459个样本，80个属性。

需要做的工作：根据训练集的属性预测测试集的房价。

# 定性分析
1 属性的意义
SalePrice 以美元出售的房产价格。 
MSSubClass 建筑类 
MSZoning 城市总体规划分区 
LotFrontage 连接物业的街道线 
LotArea: Lot size in square feet 方块大小 
Street 道路入口类型 
Alley 巷类型 
LotShape 地产的外形 
LandContour 地产的扁平化 
Utilities 地产的公用事业类型 
LotConfig 地产配置 
LandSlope 地产的坡 
Neighborhood 城市范围内的物理位置 
Condition1 接近主干道或铁路 
Condition2 接近主路或铁路 
BldgType 住宅类型 
HouseStyle 居家风格 
OverallQual 整体质量和表面质量 
OverallCond 总体状态额定值 
YearBuilt 原施工日期 
YearRemodAdd 重塑日期 
RoofStyle 屋顶类型 
RoofMatl 屋顶材料 
Exterior1st 房屋外墙 
Exterior2nd 外部第二层：房屋外部覆盖物 
MasVnrType 圬工单板型 
MasVnrArea 砌体单板覆盖面积 
ExterQual: 外观材质 
ExterCond 外墙材料的现状 
Foundation 地基类型 
BsmtQual 地下室的高度 
BsmtCond 地下室概况 
BsmtExposure: 走道或花园式地下室墙 
BsmtFinType1 地下室竣工面积质量 
BsmtFinSF1 1型成品面积 
BsmtFinType2 第二成品区域的质量（如果存在） 
BsmtFinSF2 2型成品面积 
BsmtUnfSF 地下室面积 
TotalBsmtSF 地下室面积总计面积 
Heating 暖气方式 
HeatingQC 暖气质量与条件 
CentralAir 空调 
Electrical 电气系统 
1stFlrSF 一楼面积 
2ndFlrSF 二楼面积 
LowQualFinSF 低质量完工面积（所有楼层） 
GrLivArea 高档（地面）居住面积 
BsmtFullBath 地下室全浴室 
BsmtHalfBath 地下室半浴室 
FullBath 高档浴室 
HalfBath 半日以上洗澡浴室 
Bedroom 地下室层以上的卧室数 
Kitchen 厨房数量 
KitchenQual 厨房品质 
TotRmsAbvGrd 总房间（不包括浴室） 
Functional 家庭功能评级 
Fireplaces 壁炉数 
FireplaceQu 壁炉质量 
GarageType 车库位置 
GarageYrBlt 车库建成年 
GarageFinish 车库的内饰 
GarageCars 车库容量大小 
GarageArea 车库大小 
GarageQual 车库质量 
GarageCond 车库状况 
PavedDrive 铺好的车道 
WoodDeckSF 木制甲板面积 
OpenPorchSF 外部走廊面积 
EnclosedPorch 闭走廊面积 
3SsnPorch: 三季走廊面积 
ScreenPorch 屏风走廊面积 
PoolArea 泳池面积 
PoolQC 泳池的质量 
Fence 围栏质量 
MiscFeature 其他类别的杂项特征 
MiscVal 杂项价值 
MoSold 月售出 
YrSold 年销售 
SaleType 销售类型 
SaleCondition 销售条件

2 属性分析
  可以看出，标签为房价，而对于79个属性主要分为几分方面： 
（1）房子地理位置： 
MSSubClass、MSZoning、LotFrontage、LotArea、Street、Alley、LotShape、LandContour、Utilities、LotConfig、LandSlope、Neighborhood、Condition1、Condition2 
（2）房子风格： 
BldgType、HouseStyle、OverallQual、OverallCond 
（3）房子装修： 
YearBuilt、YearRemodAdd、RoofStyle、RoofMatl、Exterior1st、Exterior2nd、MasVnrType、MasVnrArea、ExterQual: 
ExterCond 
（4）地下室： 
Foundation、BsmtQual、BsmtCond、BsmtExposure:、BsmtFinType1、BsmtFinSF1、BsmtFinType2、BsmtFinSF2、BsmtUnfSF、TotalBsmtSF 
（5）冷暖气： 
Heating、HeatingQC、CentralAir、Electrical 
（6）居住面积： 
1stFlrSF、2ndFlrSF、LowQualFinSF、GrLivArea 
（7）功能房间： 
BsmtFullBath、BsmtHalfBath、FullBath、HalfBath、Bedroom、Kitchen、KitchenQual、TotRmsAbvGrd、Functional 
（8）车库： 
GarageType、GarageYrBlt、GarageFinish、GarageCars、GarageArea、GarageQual、GarageCond、PavedDrive 
（9）其他面积： 
WoodDeckSF、OpenPorchSF、EnclosedPorch、3SsnPorch:、ScreenPorch、PoolArea 
（10）销售： 
MoSold、YrSold、SaleType、SaleCondition 
（11）其他： 
Fireplaces、FireplaceQu、PoolQC、Fence、MiscFeature、MiscVal

  假如数据真实可靠，则从实际情况考虑，对于一个房子的价格，最重要的属性首先应该
	有：地理位置、面积、地下室、冷暖气、车库、房子质量，还有会影响到房价的有：销售条件如时间和方式。所以先可以着重讨论这些方面的属性。
	
# 缺失值处理
“暴力填充”：
 1.连续数据均值填充
train1=train1.fillna(train1.mean())
 2.离散型数据"None"
train1=train1.fillna('None')	

# 测试集制作
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)

# BaseLine模型的训练
xg_reg=xgb.XGBRegressor(objective='reg:linear',colsample_bytree=0.6,learning_rate=0.01,max_depth=5,alpha=10,n_estimators=3000,subsample=0.7,random_state=123)
xg_reg.fit(X_train,y_train)

# 模型评测
pred=xg_reg.predict(X_test)
rmse=np.sqrt(mean_squared_error(y_test,pred))
23161

logrmse=np.sqrt(mean_squared_error(np.log(y_test),np.log(pred)))
0.1048

# 第一轮总结：
房价预测这个题目看似简单，实质上很有难度，主要是属性很多，而且缺失值也很多，数据预处理有难度。 
本文进行数据预处理考虑还简单，可以继续考虑属性间的线性关系程度，以及属性间合并，或属性分拆等。 
对于结果，还有很大提升空间。



	
	
	
	
	
	
	
	






