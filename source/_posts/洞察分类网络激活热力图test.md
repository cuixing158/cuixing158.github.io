---
title: 洞察分类网络激活热力图
catalog: true
date: 2020-01-07 22:25:53
subtitle: 
header-img: "shenzhen_zoo.jpg"
tags: 
- Deep Visualization
- matlab
catagories: 
- matlab
- CNN
---

> 深度神经网络经常被当作“黑盒”，网络预测为什么是那样，很少有直观的解释，但是依靠CAM（Class Activation Mapping）可以可视化网络的最后一层激活值，对网络预测的类别区域进行热力图分析，看出网络到底学到了图片的那部分内容。

## squeezenet预测
为了洞察网络预测分类到底“学习”到了什么，本示例通过USB摄像头实时采集图像，分析分类网络激活类别的热力图，越红的区域代表网络学习的特征越强，以squeezenet轻量级网络为例说明。

```matlab
net =squeezenet; % come from https://ww2.mathworks.cn/matlabcentral/fileexchange/?term=authorid%3A1211826
featureLayer = 'relu_conv10';
% analyzeNetwork(net)

%%
netInputSize = net.Layers(1).InputSize(1:2);
classes = net.Layers(end).Classes;

%% capture image
camera = webcam;
img = camera.snapshot();
H = size(img,1);
W = size(img,2);
h = vision.DeployableVideoPlayer;

showImage = zeros(H,2*W,3);
h(showImage)
while h.isOpen()
    ori = camera.snapshot();
    img = imresize(ori,netInputSize);
    predictScores = predict(net,img);
    [maxScores,indxs] = maxk(predictScores,3,2);%最大3个类别分数，沿着列的方向取最大值
    predictLabel = classes(indxs);
    
    features = activations(net,img,featureLayer);
    extractFeaturesMap = features(:,:,indxs);
    projectFeatures = postProcess(extractFeaturesMap,[H,W]);
    projectFeatures(projectFeatures<0.2)=0;
    normalizeImg = im2double(ori);
%     imagesc('CData',projectFeatures,'AlphaData',0.5);
    out = imtile(projectFeatures(:,:,1));
    imshow(out);
    strongestFeatureMap = uint8(projectFeatures(:,:,1)*255);% 把特征图的数值看作强度等级，值越大表示激活的区域越强，转换为uint8是为了对应到255个颜色等级图上
    RGB = ind2rgb(strongestFeatureMap,jet(255));% RGB 范围为[0,1]内，double类型
    combineImg = normalizeImg+RGB;
    combineImg = mat2gray(combineImg);
    
    showImage(:,1:W,:) = normalizeImg;
    showImage(:,W+1:end,:) = combineImg;
    showText = sprintf('%s %.2f\n%s %.2f\n%s %.2f',string(predictLabel(1)),maxScores(1),...
        string(predictLabel(2)),maxScores(2),...
        string(predictLabel(3)),maxScores(3));
    showImage = insertText(showImage,[20,50],showText);
    h(showImage);
    
end

function projectFeatures = postProcess(featureMaps,outPutSize)
% 把特征图h*w*c归一化到[0,1]范围，然后反投影到原图大小
% 输入：featureMaps为h*w*c大小矩阵
%      outPutSize为1*2大小矩阵，表示[H,W]
% 输出：projectFeatures为H*W*c的[0,1]范围归一化反投影特征矩阵
channels = size(featureMaps,3);
for  i = 1:channels
    minv = min(featureMaps(:,:,i),[],'all');
    maxv = max(featureMaps(:,:,i),[],'all');
    featureMaps(:,:,i) = (featureMaps(:,:,i)-minv)./(maxv-minv);
end
projectFeatures = imresize(featureMaps,outPutSize,'bicubic');
end
```

![fig5](predict.png)
<center>Fig.预测热力图</center>

## googlenet预测
googlenet全连接层之前的卷积层卷积核个数并非1000，所以相较于squeezenet要进行适当变化，具体在于对每个特征通道进行一个“分数加权”，计算出适当的特征图，然后插值到原图大小即可。对于C++版的生成，改写一些不支持的操作，也比较容易达到独立运行此应用程序的目的。改后的代码文件命名为googlenet_predict_map_coder.m，其内容为：
```matlab
function outImage = googlenet_predict_map_coder(image) %#codegen
% 用于代码生成的测试脚本程序，主要功能完成googlenet预测激活图实时显示,输入,image必须为固定大小
persistent mynet fc
if isempty(mynet)
    mynet = coder.loadDeepLearningNetwork('googlenet');
    fc = coder.load('fullyconvWB.mat');
end

% all classes
    classes = {'tench'
'goldfish'
'great white shark'
'tiger shark'
'hammerhead'
'electric ray'
'stingray'
'cock'
'hen'
'ostrich'
'brambling'
'goldfinch'
'house finch'
'junco'
'indigo bunting'
'robin'
'bulbul'
'jay'
'magpie'
'chickadee'
'water ouzel'
'kite'
'bald eagle'
'vulture'
'great grey owl'
'European fire salamander'
'common newt'
'eft'
'spotted salamander'
'axolotl'
'bullfrog'
'tree frog'
'tailed frog'
'loggerhead'
'leatherback turtle'
'mud turtle'
'terrapin'
'box turtle'
'banded gecko'
'common iguana'
'American chameleon'
'whiptail'
'agama'
'frilled lizard'
'alligator lizard'
'Gila monster'
'green lizard'
'African chameleon'
'Komodo dragon'
'African crocodile'
'American alligator'
'triceratops'
'thunder snake'
'ringneck snake'
'hognose snake'
'green snake'
'king snake'
'garter snake'
'water snake'
'vine snake'
'night snake'
'boa constrictor'
'rock python'
'Indian cobra'
'green mamba'
'sea snake'
'horned viper'
'diamondback'
'sidewinder'
'trilobite'
'harvestman'
'scorpion'
'black and gold garden spider'
'barn spider'
'garden spider'
'black widow'
'tarantula'
'wolf spider'
'tick'
'centipede'
'black grouse'
'ptarmigan'
'ruffed grouse'
'prairie chicken'
'peacock'
'quail'
'partridge'
'African grey'
'macaw'
'sulphur-crested cockatoo'
'lorikeet'
'coucal'
'bee eater'
'hornbill'
'hummingbird'
'jacamar'
'toucan'
'drake'
'red-breasted merganser'
'goose'
'black swan'
'tusker'
'echidna'
'platypus'
'wallaby'
'koala'
'wombat'
'jellyfish'
'sea anemone'
'brain coral'
'flatworm'
'nematode'
'conch'
'snail'
'slug'
'sea slug'
'chiton'
'chambered nautilus'
'Dungeness crab'
'rock crab'
'fiddler crab'
'king crab'
'American lobster'
'spiny lobster'
'crayfish'
'hermit crab'
'isopod'
'white stork'
'black stork'
'spoonbill'
'flamingo'
'little blue heron'
'American egret'
'bittern'
'crane'
'limpkin'
'European gallinule'
'American coot'
'bustard'
'ruddy turnstone'
'red-backed sandpiper'
'redshank'
'dowitcher'
'oystercatcher'
'pelican'
'king penguin'
'albatross'
'grey whale'
'killer whale'
'dugong'
'sea lion'
'Chihuahua'
'Japanese spaniel'
'Maltese dog'
'Pekinese'
'Shih-Tzu'
'Blenheim spaniel'
'papillon'
'toy terrier'
'Rhodesian ridgeback'
'Afghan hound'
'basset'
'beagle'
'bloodhound'
'bluetick'
'black-and-tan coonhound'
'Walker hound'
'English foxhound'
'redbone'
'borzoi'
'Irish wolfhound'
'Italian greyhound'
'whippet'
'Ibizan hound'
'Norwegian elkhound'
'otterhound'
'Saluki'
'Scottish deerhound'
'Weimaraner'
'Staffordshire bullterrier'
'American Staffordshire terrier'
'Bedlington terrier'
'Border terrier'
'Kerry blue terrier'
'Irish terrier'
'Norfolk terrier'
'Norwich terrier'
'Yorkshire terrier'
'wire-haired fox terrier'
'Lakeland terrier'
'Sealyham terrier'
'Airedale'
'cairn'
'Australian terrier'
'Dandie Dinmont'
'Boston bull'
'miniature schnauzer'
'giant schnauzer'
'standard schnauzer'
'Scotch terrier'
'Tibetan terrier'
'silky terrier'
'soft-coated wheaten terrier'
'West Highland white terrier'
'Lhasa'
'flat-coated retriever'
'curly-coated retriever'
'golden retriever'
'Labrador retriever'
'Chesapeake Bay retriever'
'German short-haired pointer'
'vizsla'
'English setter'
'Irish setter'
'Gordon setter'
'Brittany spaniel'
'clumber'
'English springer'
'Welsh springer spaniel'
'cocker spaniel'
'Sussex spaniel'
'Irish water spaniel'
'kuvasz'
'schipperke'
'groenendael'
'malinois'
'briard'
'kelpie'
'komondor'
'Old English sheepdog'
'Shetland sheepdog'
'collie'
'Border collie'
'Bouvier des Flandres'
'Rottweiler'
'German shepherd'
'Doberman'
'miniature pinscher'
'Greater Swiss Mountain dog'
'Bernese mountain dog'
'Appenzeller'
'EntleBucher'
'boxer'
'bull mastiff'
'Tibetan mastiff'
'French bulldog'
'Great Dane'
'Saint Bernard'
'Eskimo dog'
'malamute'
'Siberian husky'
'dalmatian'
'affenpinscher'
'basenji'
'pug'
'Leonberg'
'Newfoundland'
'Great Pyrenees'
'Samoyed'
'Pomeranian'
'chow'
'keeshond'
'Brabancon griffon'
'Pembroke'
'Cardigan'
'toy poodle'
'miniature poodle'
'standard poodle'
'Mexican hairless'
'timber wolf'
'white wolf'
'red wolf'
'coyote'
'dingo'
'dhole'
'African hunting dog'
'hyena'
'red fox'
'kit fox'
'Arctic fox'
'grey fox'
'tabby'
'tiger cat'
'Persian cat'
'Siamese cat'
'Egyptian cat'
'cougar'
'lynx'
'leopard'
'snow leopard'
'jaguar'
'lion'
'tiger'
'cheetah'
'brown bear'
'American black bear'
'ice bear'
'sloth bear'
'mongoose'
'meerkat'
'tiger beetle'
'ladybug'
'ground beetle'
'long-horned beetle'
'leaf beetle'
'dung beetle'
'rhinoceros beetle'
'weevil'
'fly'
'bee'
'ant'
'grasshopper'
'cricket'
'walking stick'
'cockroach'
'mantis'
'cicada'
'leafhopper'
'lacewing'
'dragonfly'
'damselfly'
'admiral'
'ringlet'
'monarch'
'cabbage butterfly'
'sulphur butterfly'
'lycaenid'
'starfish'
'sea urchin'
'sea cucumber'
'wood rabbit'
'hare'
'Angora'
'hamster'
'porcupine'
'fox squirrel'
'marmot'
'beaver'
'guinea pig'
'sorrel'
'zebra'
'hog'
'wild boar'
'warthog'
'hippopotamus'
'ox'
'water buffalo'
'bison'
'ram'
'bighorn'
'ibex'
'hartebeest'
'impala'
'gazelle'
'Arabian camel'
'llama'
'weasel'
'mink'
'polecat'
'black-footed ferret'
'otter'
'skunk'
'badger'
'armadillo'
'three-toed sloth'
'orangutan'
'gorilla'
'chimpanzee'
'gibbon'
'siamang'
'guenon'
'patas'
'baboon'
'macaque'
'langur'
'colobus'
'proboscis monkey'
'marmoset'
'capuchin'
'howler monkey'
'titi'
'spider monkey'
'squirrel monkey'
'Madagascar cat'
'indri'
'Indian elephant'
'African elephant'
'lesser panda'
'giant panda'
'barracouta'
'eel'
'coho'
'rock beauty'
'anemone fish'
'sturgeon'
'gar'
'lionfish'
'puffer'
'abacus'
'abaya'
'academic gown'
'accordion'
'acoustic guitar'
'aircraft carrier'
'airliner'
'airship'
'altar'
'ambulance'
'amphibian'
'analog clock'
'apiary'
'apron'
'ashcan'
'assault rifle'
'backpack'
'bakery'
'balance beam'
'balloon'
'ballpoint'
'Band Aid'
'banjo'
'bannister'
'barbell'
'barber chair'
'barbershop'
'barn'
'barometer'
'barrel'
'barrow'
'baseball'
'basketball'
'bassinet'
'bassoon'
'bathing cap'
'bath towel'
'bathtub'
'beach wagon'
'beacon'
'beaker'
'bearskin'
'beer bottle'
'beer glass'
'bell cote'
'bib'
'bicycle-built-for-two'
'bikini'
'binder'
'binoculars'
'birdhouse'
'boathouse'
'bobsled'
'bolo tie'
'bonnet'
'bookcase'
'bookshop'
'bottlecap'
'bow'
'bow tie'
'brass'
'brassiere'
'breakwater'
'breastplate'
'broom'
'bucket'
'buckle'
'bulletproof vest'
'bullet train'
'butcher shop'
'cab'
'caldron'
'candle'
'cannon'
'canoe'
'can opener'
'cardigan'
'car mirror'
'carousel'
'carpenters kit'
'carton'
'car wheel'
'cash machine'
'cassette'
'cassette player'
'castle'
'catamaran'
'CD player'
'cello'
'cellular telephone'
'chain'
'chainlink fence'
'chain mail'
'chain saw'
'chest'
'chiffonier'
'chime'
'china cabinet'
'Christmas stocking'
'church'
'cinema'
'cleaver'
'cliff dwelling'
'cloak'
'clog'
'cocktail shaker'
'coffee mug'
'coffeepot'
'coil'
'combination lock'
'computer keyboard'
'confectionery'
'container ship'
'convertible'
'corkscrew'
'cornet'
'cowboy boot'
'cowboy hat'
'cradle'
'crane (machine)'
'crash helmet'
'crate'
'crib'
'Crock Pot'
'croquet ball'
'crutch'
'cuirass'
'dam'
'desk'
'desktop computer'
'dial telephone'
'diaper'
'digital clock'
'digital watch'
'dining table'
'dishrag'
'dishwasher'
'disk brake'
'dock'
'dogsled'
'dome'
'doormat'
'drilling platform'
'drum'
'drumstick'
'dumbbell'
'Dutch oven'
'electric fan'
'electric guitar'
'electric locomotive'
'entertainment center'
'envelope'
'espresso maker'
'face powder'
'feather boa'
'file'
'fireboat'
'fire engine'
'fire screen'
'flagpole'
'flute'
'folding chair'
'football helmet'
'forklift'
'fountain'
'fountain pen'
'four-poster'
'freight car'
'French horn'
'frying pan'
'fur coat'
'garbage truck'
'gasmask'
'gas pump'
'goblet'
'go-kart'
'golf ball'
'golfcart'
'gondola'
'gong'
'gown'
'grand piano'
'greenhouse'
'grille'
'grocery store'
'guillotine'
'hair slide'
'hair spray'
'half track'
'hammer'
'hamper'
'hand blower'
'hand-held computer'
'handkerchief'
'hard disc'
'harmonica'
'harp'
'harvester'
'hatchet'
'holster'
'home theater'
'honeycomb'
'hook'
'hoopskirt'
'horizontal bar'
'horse cart'
'hourglass'
'iPod'
'iron'
'jack antern'
'jean'
'jeep'
'jersey'
'jigsaw puzzle'
'jinrikisha'
'joystick'
'kimono'
'knee pad'
'knot'
'lab coat'
'ladle'
'lampshade'
'laptop'
'lawn mower'
'lens cap'
'letter opener'
'library'
'lifeboat'
'lighter'
'limousine'
'liner'
'lipstick'
'Loafer'
'lotion'
'loudspeaker'
'loupe'
'lumbermill'
'magnetic compass'
'mailbag'
'mailbox'
'maillot'
'maillot, tank suit'
'manhole cover'
'maraca'
'marimba'
'mask'
'matchstick'
'maypole'
'maze'
'measuring cup'
'medicine chest'
'megalith'
'microphone'
'microwave'
'military uniform'
'milk can'
'minibus'
'miniskirt'
'minivan'
'missile'
'mitten'
'mixing bowl'
'mobile home'
'Model T'
'modem'
'monastery'
'monitor'
'moped'
'mortar'
'mortarboard'
'mosque'
'mosquito net'
'motor scooter'
'mountain bike'
'mountain tent'
'mouse'
'mousetrap'
'moving van'
'muzzle'
'nail'
'neck brace'
'necklace'
'nipple'
'notebook'
'obelisk'
'oboe'
'ocarina'
'odometer'
'oil filter'
'organ'
'oscilloscope'
'overskirt'
'oxcart'
'oxygen mask'
'packet'
'paddle'
'paddlewheel'
'padlock'
'paintbrush'
'pajama'
'palace'
'panpipe'
'paper towel'
'parachute'
'parallel bars'
'park bench'
'parking meter'
'passenger car'
'patio'
'pay-phone'
'pedestal'
'pencil box'
'pencil sharpener'
'perfume'
'Petri dish'
'photocopier'
'pick'
'pickelhaube'
'picket fence'
'pickup'
'pier'
'piggy bank'
'pill bottle'
'pillow'
'ping-pong ball'
'pinwheel'
'pirate'
'pitcher'
'plane'
'planetarium'
'plastic bag'
'plate rack'
'plow'
'plunger'
'Polaroid camera'
'pole'
'police van'
'poncho'
'pool table'
'pop bottle'
'pot'
'potter wheel'
'power drill'
'prayer rug'
'printer'
'prison'
'projectile'
'projector'
'puck'
'punching bag'
'purse'
'quill'
'quilt'
'racer'
'racket'
'radiator'
'radio'
'radio telescope'
'rain barrel'
'recreational vehicle'
'reel'
'reflex camera'
'refrigerator'
'remote control'
'restaurant'
'revolver'
'rifle'
'rocking chair'
'rotisserie'
'rubber eraser'
'rugby ball'
'rule'
'running shoe'
'safe'
'safety pin'
'saltshaker'
'sandal'
'sarong'
'sax'
'scabbard'
'scale'
'school bus'
'schooner'
'scoreboard'
'screen'
'screw'
'screwdriver'
'seat belt'
'sewing machine'
'shield'
'shoe shop'
'shoji'
'shopping basket'
'shopping cart'
'shovel'
'shower cap'
'shower curtain'
'ski'
'ski mask'
'sleeping bag'
'slide rule'
'sliding door'
'slot'
'snorkel'
'snowmobile'
'snowplow'
'soap dispenser'
'soccer ball'
'sock'
'solar dish'
'sombrero'
'soup bowl'
'space bar'
'space heater'
'space shuttle'
'spatula'
'speedboat'
'spider web'
'spindle'
'sports car'
'spotlight'
'stage'
'steam locomotive'
'steel arch bridge'
'steel drum'
'stethoscope'
'stole'
'stone wall'
'stopwatch'
'stove'
'strainer'
'streetcar'
'stretcher'
'studio couch'
'stupa'
'submarine'
'suit'
'sundial'
'sunglass'
'sunglasses'
'sunscreen'
'suspension bridge'
'swab'
'sweatshirt'
'swimming trunks'
'swing'
'switch'
'syringe'
'table lamp'
'tank'
'tape player'
'teapot'
'teddy'
'television'
'tennis ball'
'thatch'
'theater curtain'
'thimble'
'thresher'
'throne'
'tile roof'
'toaster'
'tobacco shop'
'toilet seat'
'torch'
'totem pole'
'tow truck'
'toyshop'
'tractor'
'trailer truck'
'tray'
'trench coat'
'tricycle'
'trimaran'
'tripod'
'triumphal arch'
'trolleybus'
'trombone'
'tub'
'turnstile'
'typewriter keyboard'
'umbrella'
'unicycle'
'upright'
'vacuum'
'vase'
'vault'
'velvet'
'vending machine'
'vestment'
'viaduct'
'violin'
'volleyball'
'waffle iron'
'wall clock'
'wallet'
'wardrobe'
'warplane'
'washbasin'
'washer'
'water bottle'
'water jug'
'water tower'
'whiskey jug'
'whistle'
'wig'
'window screen'
'window shade'
'Windsor tie'
'wine bottle'
'wing'
'wok'
'wooden spoon'
'wool'
'worm fence'
'wreck'
'yawl'
'yurt'
'web site'
'comic book'
'crossword puzzle'
'street sign'
'traffic light'
'book jacket'
'menu'
'plate'
'guacamole'
'consomme'
'hot pot'
'trifle'
'ice cream'
'ice lolly'
'French loaf'
'bagel'
'pretzel'
'cheeseburger'
'hotdog'
'mashed potato'
'head cabbage'
'broccoli'
'cauliflower'
'zucchini'
'spaghetti squash'
'acorn squash'
'butternut squash'
'cucumber'
'artichoke'
'bell pepper'
'cardoon'
'mushroom'
'Granny Smith'
'strawberry'
'orange'
'lemon'
'fig'
'pineapple'
'banana'
'jackfruit'
'custard apple'
'pomegranate'
'hay'
'carbonara'
'chocolate sauce'
'dough'
'meat loaf'
'pizza'
'potpie'
'burrito'
'red wine'
'espresso'
'cup'
'eggnog'
'alp'
'bubble'
'cliff'
'coral reef'
'geyser'
'lakeside'
'promontory'
'sandbar'
'seashore'
'valley'
'volcano'
'ballplayer'
'groom'
'scuba diver'
'rapeseed'
'daisy'
'yellow lady slipper'
'corn'
'acorn'
'hip'
'buckeye'
'coral fungus'
'agaric'
'gyromitra'
'stinkhorn'
'earthstar'
'hen-of-the-woods'
'bolete'
'ear'
'toilet tissue'}; %mynet.Layers(end).Classes;
netInputSize = [224,224]; %mynet.Layers(1).InputSize(1:2);
H = 480;W = 640;
RGB = zeros(H,W,3);

image = imresize(image,[H,W]);
showImage = zeros(H,2*W,3);

% pass in input
ori = image;
featureLayer = 'inception_5b-output';
img = imresize(ori,netInputSize);
predictScores = predict(mynet,img);
[maxScores,indxs] = maxk(predictScores,3,2);%最大3个类别分数，沿着列的方向取最大值
predictLabel = cell(length(indxs),1);
for id = 1:length(indxs)
    ind = indxs(id);
    predictLabel{id} = classes{ind};
end

features = activations(mynet,img,featureLayer);
scores = squeeze(mean(features,[1,2]));

fcWeights = fc.fcWeights; % mynet.Layers(end-2).Weights;
fcBias = fc.fcBias; % mynet.Layers(end-2).Bias;
scores =  fcWeights*scores + fcBias;

[~,classIds] = maxk(scores,3);
fcdim=size(features,3);
weightVector = reshape(fcWeights(classIds(1),:),1,1,fcdim);
weightMatrix = zeros(size(features),'like',features);
for i = 1:fcdim
    weightMatrix(:,:,i) = repmat(weightVector(:,:,i),size(features,1),size(features,2));
end

classActivationMap = sum(features.*weightMatrix,3);
classActivationMap = rescale(classActivationMap);
projectFeatures = imresize(classActivationMap,[H,W],'bicubic');
projectFeatures(projectFeatures<0.2)=0;

strongestFeatureMap = uint8(projectFeatures*255);% 把特征图的数值看作强度等级，值越大表示激活的区域越强，转换为uint8是为了对应到255个颜色等级图上
RGB = ind2rgb(strongestFeatureMap,fc.jetColormap);% RGB 范围为[0,1]内，double类型
normalizeImg = im2double(ori);
combineImg = normalizeImg+RGB;
combineImg = rescale(combineImg); % [0,1]范围

showImage(:,1:W,:) = normalizeImg;
showImage(:,W+1:end,:) = combineImg;
showText = sprintf('%s %.2f\n%s %.2f\n%s %.2f',predictLabel{1},maxScores(1),...
    predictLabel{2},maxScores(2),...
    predictLabel{3},maxScores(3));
showImage = insertText(showImage,[20,50],showText);
outImage = showImage;

```
生成独立的库文件，配置生成代码：
```matlab
cfg = coder.gpuConfig('dll');
cfg.TargetLang = 'C++';
cfg.DeepLearningConfig = coder.DeepLearningConfig('cudnn'); 
cfg.DeepLearningConfig.DataType = 'fp32';
codegen -args {ones(480,640,3,'uint8')} -config cfg googlenet_predict_map_coder;
```
稍等片刻，即可生成一些源文件和库文件<br>
![fig5](gpu_coder.jpg)
<center>Fig.C++源代码</center>

## 独立的C++代码嵌入到应用程序
上面生成的源代码文件较多，但用到有效的比较少，仅需模型二进制权重文件和一些C++头文件，静态库lib，动态库dll即可，其他中间文件忽略。在C++中写好传入函数输入输出接口即可，下面C++代码是opencv与生成的代码融合的一种方式，也是比较容易实现。<br>
```C++
//
// File: main.cu
//
// GPU Coder version                    : 1.5
// CUDA/C/C++ source code generated on  : 31-May-2020 17:32:07
//

//***********************************************************************
// This automatically generated example CUDA main file shows how to call
// entry-point functions that MATLAB Coder generated. You must customize
// this file for your application. Do not modify this file directly.
// Instead, make a copy of this file, modify it, and integrate it into
// your development environment.
//
// This file initializes entry-point function arguments to a default
// size and value before calling the entry-point functions. It does
// not store or use any values returned from the entry-point functions.
// If necessary, it does pre-allocate memory for returned values.
// You can use this file as a starting point for a main function that
// you can deploy in your application.
//
// After you copy the file, and before you deploy it, you must make the
// following changes:
// * For variable-size function arguments, change the example sizes to
// the sizes that your application requires.
// * Change the example values of function arguments to the values that
// your application requires.
// * If the entry-point functions return values, store these values or
// otherwise use them as required by your application.
//
//***********************************************************************

#include <iostream>
#include <fstream>
#include <string>
#include"opencv2/opencv.hpp"

// Include Files
#include "DeepLearningNetwork.h"
#include "googlenet_predict_map_coder.h"
#include "googlenet_predict_map_coder_terminate.h"
#include "rt_nonfinite.h"

// Function Declarations
static void argInit_480x640x3_uint8_T(cv::Mat&image, char result[921600]);
static unsigned char argInit_uint8_T();
static void main_googlenet_predict_map_coder();

// Function Definitions

// Arguments    : unsigned char result[921600]
// Return Type  : void
//
static void argInit_480x640x3_uint8_T(cv::Mat&image, unsigned char result[921600])
{
	if (image.size() != cv::Size(640, 480))
	{
		cv::resize(image, image, cv::Size(640, 480));
	}

	// Loop over the array to initialize each element.
	for (int idx0 = 0; idx0 < 480; idx0++) {
		unsigned char* data = image.ptr<unsigned char>(idx0);
		for (int idx1 = 0; idx1 < 640; idx1++) {
			for (int idx2 = 0; idx2 < 3; idx2++) {
				// Set the value of the array element.
				// Change this value to the value that the application requires.
					result[(idx0 + 480 * idx1) + 307200 * idx2] = data[idx2];
			}
			data += 3;
		}
	}
}

//
// Arguments    : void
// Return Type  : unsigned char
//
static unsigned char argInit_uint8_T()
{
	return 0U;
}

//
// Arguments    : void
// Return Type  : void
//
static void main_googlenet_predict_map_coder()
{
	static double outImage[1843200];
	static unsigned char b[921600];

	// Initialize function 'googlenet_predict_map_coder' input arguments.
	// Initialize function input argument 'image'.
	// Call the entry-point 'googlenet_predict_map_coder'.
	cv::VideoCapture cap(0);
	cv::Mat ori, image;
	if (!cap.read(ori))
	{
		std::cerr << "can't open camera!" << std::endl;
	}
	while (cap.read(ori))
	{
		if (ori.empty())
		{
			std::cerr << "can't read image!" << std::endl;
		}
		image = ori.clone();
		cv::Mat ori2 = ori.clone();

		//GPU coder代码，不占用CPU，但占用很少一部分GPU，耗时6ms左右
		double t1 = cv::getTickCount();
		argInit_480x640x3_uint8_T(image, b);
		googlenet_predict_map_coder(b, outImage);
		double t2 = cv::getTickCount();
		printf("googlenet_predict take time:%.2f ms\n", (t2 - t1) / cv::getTickFrequency() * 1000);

		//outImage 转换为Opencv Mat
		cv::Mat outCVImage = cv::Mat( 480, 2 * 640, CV_64FC3,cv::Scalar(0,0,0));
		for (int i = 0; i < outCVImage.rows; i++)
		{
			cv::Vec3d* data = outCVImage.ptr<cv::Vec3d>(i);
			for (int j = 0; j < outCVImage.cols; j++)
			{
				for (int m = 0; m < outCVImage.channels(); m++)
				{
						data[j][m] = outImage[(i + 480 * j) + 480*640*2 * m];
				}
			}
		}
		
		cv::Mat dst;
		outCVImage.convertTo(dst, CV_32FC3, 1.0);
		cv::imshow("outCVImage", dst);
		int key = cv::waitKey(1);
		if (key == 27)
		{
			break;
		}
		if (key == ' ')
		{
			cv::waitKey();
		}

	}
}

//
// Arguments    : int argc
//                const char * const argv[]
// Return Type  : int
//
int main(int, const char * const[])
{
	// The initialize function is being called automatically from your entry-point function. So, a call to initialize is not included here. 
	// Invoke the entry-point functions.
	// You can call entry-point functions multiple times.
	main_googlenet_predict_map_coder();

	// Terminate the application.
	// You do not need to do this more than one time.
	googlenet_predict_map_coder_terminate();
	return 0;
}

//
// File trailer for main.cu
//
// [EOF]
//
```
![fig5](predict_opencv.jpg)
<center>Fig.嵌入到独立应用的预测热力图</center>
