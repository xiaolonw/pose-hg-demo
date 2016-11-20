

require 'paths'
require 'nngraph'
require 'cunn'
require 'cudnn'
local ffi = require 'ffi'

paths.dofile('util.lua')
paths.dofile('img.lua')


m = torch.load('umich-stacked-hourglass.t7')
list_file = '/nfs.yoda/xiaolonw/torch_projects/my-eyescream/affordance3/pose_estimation/makelist/friends_matches_testposelist.txt'
path_dataset = '/scratch/xiaolonw/affordance_general/data2/'
savepath = './samples/'


local testnum = 2834
local maxPathLength = 300
local loadSize = 256

lblset = torch.Tensor(testnum, 34):float():fill(0)
imagePath = torch.CharTensor()
imagePath:resize(testnum, maxPathLength):fill(0)

local s_data = imagePath:data()


f = assert(io.open(list_file, "r"))
for i = 1, testnum do 

  -- get name
  list = f:read("*line")
  cnt = 0 
  for str in string.gmatch(list, "%S+") do
    -- print(str)
    cnt = cnt + 1
    if cnt == 1 then 
      filename = str
    else
      lblset[{{i}, {cnt - 1} }] = tonumber(str)
    end

  end
  assert(cnt == 35)

  filename = path_dataset .. filename 
  ffi.copy(s_data, filename)
  s_data = s_data + maxPathLength


  if i % 1000 == 0 then
    print(i)
    print(ffi.string(torch.data(self.imagePath[i])))
    -- print(ffi.string(torch.data(self.labelPath[i])) )

  end
  count = count + 1

end

f:close()


for i = 1, testnum do 

	print(i)

	local imname = ffi.string(torch.data(imagePath[i]))
	local im = image.load(imname)

	local x_array = torch.Tensor(17):float()
	local y_array = torch.Tensor(17):float()
	local now_array = lblset[i]

	local height = image:size()[2]
	local width  = image:size()[3]

	for j = 1, 17 do 
		x_array[j] = now_array[j * 2 - 1]
		y_array[j] = now_array[j * 2 ]
	end

	local minx = math.max(x_array:min() - 10, 1)
	local miny = math.max(y_array:min() - 10, 1)
	local maxx = math.min(x_array:max() + 10, width)
	local maxy = math.min(y_array:max() + 10, height)

	local pose_height = ( maxy - miny )
	local pose_width  = ( maxx - minx )

	local ratioh = pose_height / 256.0
	local ratiow = pose_width  / 256.0

 
	local crop_img = im[{{}, {miny, maxy}, {minx, maxx}}] 
	local inp = image.scale(cropimg, loadSize, loadSize)

	local out = m:forward(inp:view(1,3,256,256):cuda())
    cutorch.synchronize()
    local hm = out[2][1]:float()
    hm[hm:lt(0)] = 0

    local fake_center = torch.Tensor(2)
    fake_center[1] = 300
    fake_center[2] = 300

    local preds_hm, preds_img = getPreds(hm, fake_center, 1.0)

    preds_hm:mul(4)
    local dispImg = drawOutput(inp, hm, preds_hm[1])
    local imgname = paths.concat(savepath, string.format('%04d.jpg', i ))

    image.save(imgname, dispImg )


    collectgarbage()







end























