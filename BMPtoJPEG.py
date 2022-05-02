import time
from PIL import Image
import numpy as np
from scipy.fftpack import dct
import sys
import JpegStandard
from bitstream import BitStream


DEBUG = False
"""
Global Variable / Parameters
"""
#DEBUG = True
bmpTestPathName = 'test.bmp'  # DEBUG use file ''
JpegTestFileName = 'test.jpg'

LUMINANCE = 0
CHROMINANCE = 1


"""
DCT Function
"""
# 計算 DCT 系數
def dct_c(u):
    return np.sqrt(1/8) if u == 0 else np.sqrt(2/8)

# 計算 DCT
def dct2_Num():
    martix = np.zeros([8, 8])
    for i in range(8):
        for j in range(8):
            martix[i, j] = dct_c(i) * np.cos(np.pi * i * ((2 * j + 1)) / 16)
    return martix


# 調用 DCT 計算, 返回一個矩陣
def dct2(martix):
    numMartix = dct2_Num()
    res = np.dot(numMartix, martix)
    res = np.dot(res, np.transpose(numMartix))
    return res  

"""
Hex to Bytes
"""

# 將 16 進制轉換成 Byte
def hexToBytes(hexStr):
    hexLen = len(hexStr) // 2
    tmp = np.zeros([hexLen], dtype=np.int32)
    for i in range(hexLen):
        tmp[i] = int(hexStr[2 * i: 2 * i + 2], 16)
    tmp = tmp.tolist()
    tmp = bytes(tmp)
    return tmp

"""
bit string to Stream
"""
# 將 bit 字符串轉換為 Stream 輸入格式, 是用來把 bit 字符串輸入到 BitStream 的中間件
def bitStrToStream(bitStr):
    return [(c == '1') for c in bitStr]

"""
DC 編碼
"""
# 對 DC 進行編碼, 並對其用 JPEG 哈夫曼表編碼
def encodeDC(DCData, DCType):
    DCEncoded = ''
    # DC 大小(位長)編碼
    size = int(DCData).bit_length()
    if(DCType == LUMINANCE):
        DCEncoded += JpegStandard.DCLuminanceSizeEncode[size]
    elif(DCType == CHROMINANCE):
        DCEncoded += JpegStandard.DCChrominanceSizeEncode[size]
    # DC 數據編碼
    dataEncoded = ''
    if(DCData > 0):
        dataEncoded = bin(DCData)[2:]
    else:
        dataEncoded = ''.join([( '0' if c == '1' else '1') for c in bin(DCData)[3:]])
        # print(dataEncoded)
    DCEncoded += dataEncoded

    #if DEBUG: print(f'DC:{DCData}, size ={size} encode = {DCEncoded}')
    return DCEncoded


"""
AC 編碼
"""

# 對 AC 進行游長編碼(RLC), 並對其用 JPEG 哈夫曼表編碼
def encodeAC(ACMatrix, ACType):
    ACEncoded = ''
    ACMatrixLength = np.size(ACMatrix)
    i = 0
    while i < ACMatrixLength:
        runLength = 0
        for j in range(i, ACMatrixLength):
            if (ACMatrix[j] != 0):
                break
            # 在AC矩陣最後為0時編碼
            if(j == ACMatrixLength - 1):
                if(ACType == LUMINANCE):
                    ACEncoded += JpegStandard.ACLuminanceSizeEncode['00']
                else:
                    ACEncoded += JpegStandard.ACChrominanceSizeEncode['00']
                if DEBUG: print(f'E08, {i}, {ACEncoded}')
        # 到結尾全為 0 時退出
        if(j >= ACMatrixLength - 1):
            return ACEncoded

        # 記錄有多少0
        while(ACMatrix[i] == 0 and i != ACMatrixLength - 1 and runLength != 15):
            runLength += 1
            i += 1

        ACData = ACMatrix[i]
        # 二次確認, 正常情況不會出現
        if(ACData == 0 and runLength != 15):
            break
        # 編碼
        
        ACSize = int(ACData).bit_length()
        runLengthStr = str.upper(
            str(hex(runLength))[2:]) + str.upper(str(hex(ACSize))[2:])
        if(ACType == LUMINANCE):
            ACEncoded += JpegStandard.ACLuminanceSizeEncode[runLengthStr]
        else:
            ACEncoded += JpegStandard.ACChrominanceSizeEncode[runLengthStr]

        dataEncoded = ''
        if(ACData > 0):
            dataEncoded = bin(ACData)[2:]
        else:
            dataEncoded = ''.join([( '0' if c == '1' else '1') for c in bin(ACData)[3:]])
        ACEncoded += dataEncoded
        if DEBUG: print(f'ACMatrix[{i}] enocde in rlc, runLength: {runLength}, size:{ACSize} ACData:{ACData},  code:{JpegStandard.ACLuminanceSizeEncode[runLengthStr] if ACType == LUMINANCE else JpegStandard.ACChrominanceSizeEncode[runLengthStr]},{dataEncoded}')
        i += 1
    return ACEncoded


"""
Main Function
"""

# 轉換 BMP 格式圖像到 JPEG, 傳入 BMP 路徑檔案名
def BmpToJpeg(bpmFilePathName):    
    if('.bmp' not in bpmFilePathName):
        print('[ERROR] Only support .bmp file')
        exit(0)
    
    try:
        bmp = Image.open(bpmFilePathName)
        bmpWidth, bmpHeight = bmp.size[0], bmp.size[1]
        if DEBUG:
            print(
                f'BMP file: width={bmpWidth}, height={bmpHeight}, dpi={bmp.info["dpi"][0]:.2f}, imageMode={bmp.mode}')

        # 調整輸入圖像大小為 8 的倍數以確保 DCT 正常運作
        imgWidth = bmpWidth // 8 * 8 + 8 if bmpWidth % 8 else bmpWidth
        imgHeight = bmpHeight // 8 * 8 + 8 if bmpHeight % 8 else bmpHeight
        print(f'BMP new size: width={bmpWidth}, height={bmpHeight}')
        print(f'Reading image data, please wait! If your image is large, this will take a while.')
        
        # BMP 圖像數據
        bmpMatrix = np.asarray(bmp)
        imgMatrix = np.zeros((imgHeight, imgWidth, 3), dtype=np.uint8)
        for i in range(bmpHeight):
            for j in range(bmpWidth):
                imgMatrix[i][j] = bmpMatrix[i][j]
        
        print(f'Change Color Format Form RGB to YUV!')
        # 將顏色由 RGB 數據格式轉換為 YUV
        yDatas, uDatas, vDatas = Image.fromarray(
            imgMatrix).convert('YCbCr').split()
        yMatrix = np.asarray(yDatas).astype(np.int32)
        uMatrix = np.asarray(uDatas).astype(np.int32)
        vMatrix = np.asarray(vDatas).astype(np.int32)
        if DEBUG:
            print(f'Image Matrix:\nY:{yMatrix}\nU:{uMatrix}\nV:{vMatrix}')

        # DCT 計算時, 需要將矩陣數據減127, 以保證矩陣數據範圍在-128 - 127 之間
        yMatrix -= 127
        uMatrix -= 127
        vMatrix -= 127

        """
        圖像壓縮開始
        """
        print(f'Preparing to compress image!')
        imgBlockTotal = (imgHeight // 8) * (imgWidth // 8)

        # DC 矩陣和 Delta DC 矩陣
        yDCMatrix = np.zeros(imgBlockTotal, dtype=np.int32)
        uDCMatrix = np.zeros(imgBlockTotal, dtype=np.int32)
        vDCMatrix = np.zeros(imgBlockTotal, dtype=np.int32)
        dyDCMatrix = np.zeros(imgBlockTotal, dtype=np.int32)
        duDCMatrix = np.zeros(imgBlockTotal, dtype=np.int32)
        dvDCMatrix = np.zeros(imgBlockTotal, dtype=np.int32)

        # DTC中間數據是否輸出, 用於 debug.
        DCT_DEBUG = True if DEBUG and imgBlockTotal < 30 else False

        # 量化矩陣
        LuminanceQuantizationTable = JpegStandard.LuminanceQuantizationTable
        ChrominanceQuantizationTable = JpegStandard.ChrominanceQuantizationTable

        # 圖像二進制數據
        imgBitStr = ''

        imgCurrBlock = 0
        for i in range(0, imgHeight, 8):
            for j in range(0, imgWidth, 8):
                # DCT 運算
                if not DEBUG: print(f'Compression: Pixel {i, j}, Block: {imgCurrBlock}/{imgBlockTotal}, {int(imgCurrBlock/imgBlockTotal*10000)/100:.2f}%' + 50*' ', end='\r')
                yDctMatrix, uDctMatrix, vDctMatrix = dct2(
                    yMatrix[i:i + 8, j:j + 8]), dct2(uMatrix[i:i + 8, j:j + 8]), dct2(vMatrix[i:i + 8, j:j + 8])
                #yDctMatrix = dct2(yMatrix, i, j)
                #uDctMatrix = dct2(uMatrix, i, j)
                #vDctMatrix = dct2(vMatrix, i, j)
                if DCT_DEBUG:
                    print(
                        f'Y DCT Matrix:{yDctMatrix}\nU DCT Matrix:{uDctMatrix}\nV DCT Matrix:{vDctMatrix}')

                # 量化
                yQuantizationMatrix = np.around(
                    yDctMatrix / LuminanceQuantizationTable).astype(np.int32)
                uQuantizationMatrix = np.around(
                    uDctMatrix / ChrominanceQuantizationTable).astype(np.int32)
                vQuantizationMatrix = np.around(
                    vDctMatrix / ChrominanceQuantizationTable).astype(np.int32)

                if DCT_DEBUG:
                    print(
                        f'Y Quantization Matrix:{yQuantizationMatrix}\nU Quantization Matrix:{uQuantizationMatrix}\nV Quantization Matrix:{vQuantizationMatrix}')

                # 將矩陣轉換成ZigZag順序
                yZigZagMatrix = yQuantizationMatrix.reshape(
                    [64])[JpegStandard.ZigZagOrder]
                uZigZagMatrix = uQuantizationMatrix.reshape(
                    [64])[JpegStandard.ZigZagOrder]
                vZigZagMatrix = vQuantizationMatrix.reshape(
                    [64])[JpegStandard.ZigZagOrder]
                if DCT_DEBUG: print(f'Y zz Matrix:{yZigZagMatrix}\nU zz Matrix:{uZigZagMatrix}\nV zz Matrix:{vZigZagMatrix}')

                # DPCM 計算
                yDCMatrix[imgCurrBlock] = yZigZagMatrix[0]
                uDCMatrix[imgCurrBlock] = uZigZagMatrix[0]
                vDCMatrix[imgCurrBlock] = vZigZagMatrix[0]

                #print(f'Y dc Matrix:{yDCMatrix}\nU dc Matrix:{uDCMatrix}\nV cd Matrix:{vDCMatrix}')
                if(imgCurrBlock == 0):
                    dyDCMatrix[imgCurrBlock] = yDCMatrix[imgCurrBlock]
                    duDCMatrix[imgCurrBlock] = uDCMatrix[imgCurrBlock]
                    dvDCMatrix[imgCurrBlock] = vDCMatrix[imgCurrBlock]
                else:
                    dyDCMatrix[imgCurrBlock] = yDCMatrix[imgCurrBlock] - \
                        yDCMatrix[imgCurrBlock - 1]
                    duDCMatrix[imgCurrBlock] = uDCMatrix[imgCurrBlock] - \
                        uDCMatrix[imgCurrBlock - 1]
                    dvDCMatrix[imgCurrBlock] = vDCMatrix[imgCurrBlock] - \
                        vDCMatrix[imgCurrBlock - 1]
                if DCT_DEBUG:
                    print(
                        f'dY DC Matrix:{dyDCMatrix}\ndU DC Matrix:{duDCMatrix}\ndV DC Matrix:{dvDCMatrix}')
                """
                編碼
                """
                imgBitStr += encodeDC(dyDCMatrix[imgCurrBlock], LUMINANCE)
                #if DEBUG: print("Bit Str", imgBitStr)
                imgBitStr += encodeAC(yZigZagMatrix[1:], LUMINANCE)
                #if DEBUG: print("Bit Str", imgBitStr)
                imgBitStr += encodeDC(duDCMatrix[imgCurrBlock], CHROMINANCE)
                #if DEBUG: print("Bit Str", imgBitStr)
                imgBitStr += encodeAC(uZigZagMatrix[1:], CHROMINANCE)
                #if DEBUG: print("Bit Str", imgBitStr)
                imgBitStr += encodeDC(dvDCMatrix[imgCurrBlock], CHROMINANCE)
                #if DEBUG: print("Bit Str", imgBitStr)
                imgBitStr += encodeAC(vZigZagMatrix[1:], CHROMINANCE)

                if DEBUG: print("Bit Str", imgBitStr)
                # 完成一個 block
                imgCurrBlock += 1

        """
        輸出
        """
        # 输出文件位置
        if(DEBUG == False):
            JpegFileName = bpmFilePathName +'.jpg'
        else:
            JpegFileName = JpegTestFileName
        jpegFile = open(JpegFileName, 'wb+')

        # JPEG 元數據
        jpegFile.write(hexToBytes('FFD8FFE000104A46494600010100000100010000'))

        # 亮度量化表
        jpegFile.write(hexToBytes('FFDB004300'))
        LuminanceQuantizationTable = LuminanceQuantizationTable.reshape([64])
        jpegFile.write(bytes(LuminanceQuantizationTable.tolist()))

        # 色度量化表
        jpegFile.write(hexToBytes('FFDB004301'))
        ChrominanceQuantizationTable = ChrominanceQuantizationTable.reshape([64])
        jpegFile.write(bytes(ChrominanceQuantizationTable.tolist()))

        # 圖像大小
        jpegFile.write(hexToBytes('FFC0001108'))
        heightHex = hex(imgHeight)[2:]
        while len(heightHex) != 4:
            heightHex = '0' + heightHex
        jpegFile.write(hexToBytes(heightHex))
        widthHex = hex(imgWidth)[2:]
        while len(widthHex) != 4:
            widthHex = '0' + widthHex
        jpegFile.write(hexToBytes(widthHex))

        # 子采樣
        jpegFile.write(hexToBytes('03011100021101031101'))

        # 哈夫曼樹
        jpegFile.write(hexToBytes('FFC401A20000000701010101010000000000000000040503020601000708090A0B0100020203010101010100000000000000010002030405060708090A0B1000020103030204020607030402060273010203110400052112314151061361227181143291A10715B14223C152D1E1331662F0247282F12543345392A2B26373C235442793A3B33617546474C3D2E2082683090A181984944546A4B456D355281AF2E3F3C4D4E4F465758595A5B5C5D5E5F566768696A6B6C6D6E6F637475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F82939495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA110002020102030505040506040803036D0100021103042112314105511361220671819132A1B1F014C1D1E1234215526272F1332434438216925325A263B2C20773D235E2448317549308090A18192636451A2764745537F2A3B3C32829D3E3F38494A4B4C4D4E4F465758595A5B5C5D5E5F5465666768696A6B6C6D6E6F6475767778797A7B7C7D7E7F738485868788898A8B8C8D8E8F839495969798999A9B9C9D9E9F92A3A4A5A6A7A8A9AAABACADAEAFA'))
        dataLength = len(imgBitStr)

        # 長度不能除儘 8 時補至 8 位
        imgBitStr += (8 - dataLength % 8) * '1'

        jpegFile.write(hexToBytes('FFDA000C03010002110311003F00'))
        # 圖像數據
        dataBitStream = BitStream()
        dataBitStream.write(bitStrToStream(imgBitStr))
        if DEBUG: print("data stream", dataBitStream)
        dataBytes = dataBitStream.read(bytes)
        for i in range(len(dataBytes)):
            jpegFile.write(bytes([dataBytes[i]]))
            if(dataBytes[i] == 255):
                jpegFile.write(bytes([0]))
        # 結束
        jpegFile.write(hexToBytes('FFD9'))
        jpegFile.close()
        print(f'Compression: Finish!' + 50*' ')
    except IOError as err:
        print(str(err))

if __name__ == '__main__':
    if(DEBUG == False):
        if(len(sys.argv) == 1):
            print('Please Input Bmp File Path Name, you can run program with this parameters. i.e.: "./test.bmp", "C:\img\img.bmp"')
            bpmFilePathName = input()
        else:
            bpmFilePathName = sys.argv[1]
    else:
        bpmFilePathName = bmpTestPathName
    start = time.time()
    BmpToJpeg(bpmFilePathName)
    end = time.time()
    print(f'Running Time: {(end - start):.2f} second')
