import cv2
import numpy as np
import struct

outfile = 'output.gif'
infile = 'felix.mp4'

def readvid():
    '''Open input video and get bytestreams.'''
    cap = cv2.VideoCapture(infile)

    # do first frame	
    ret, frame = cap.read()
    B,G,R = cv2.split(frame)
    h,w = frame.shape[:2]
    writetitle(w,h)
    npr = np.asanyarray(R)
    npg = np.asanyarray(G)
    npb = np.asanyarray(B)
    quantr = parsecol(npr, 'r')
    quantg = parsecol(npg, 'g')
    quantb = parsecol(npb, 'b')
    s = np.unique([quantr,quantg,quantb])
    gcoltbl = makecolourtbl(s)
    writeGCE()
    writeimagedescriptor(w,h)
    writeimgdata(quantr,quantg,quantb, gcoltbl)
    counter = 1
	#read frames and display them
#	while True:
#		ret, frame = cap.read()
# 		if frame is not None:
# 			if counter == 30:
# 				h,w = frame.shape[:2]
# 				B,G,R = cv2.split(frame)
# 				cv2.imshow('video', frame)
# 				writeGCE()
# 				writeimagedescriptor(w,h)
# 				writeimgdata(R,G,B, gcoltbl)
# 				counter = 0
# 			else:
# 				counter += 1
# 			
# 		else:
# 			break
	
#		if cv2.waitKey(1) & 0xff == ord('q'):
#			break


    #stop reading video file and close display
    cap.release()
    cv2.destroyAllWindows()

	# end the file
# 	with open(outfile,'ab') as out:
#		b = bytearray()
#		b.append(59)
#		out.write(b)
#		out.close()

def mycodify(r,g,b,tbl):
	'''Convert r,g,b channels into colour codes.'''
	#print(tbl)
	return ((tbl[:,0] == r) & (tbl[:,1] == g) & (tbl[:,2] == b)).nonzero()[0][0]
vmycodify = np.vectorize(mycodify, excluded=['tbl'])

def writeimgdata(r,g,b,tbl):
	'''Perform LZW compression and append to image file.'''
	# https://www.matthewflickinger.com/lab/whatsinagif/lzw_image_data.asp
	# https://stackoverflow.com/questions/25823608/find-matching-rows-in-2-dimensional-numpy-array
	
	n = len(tbl)-1	# codes for actual table
	CC = n+1		# n+1st = clear code
	EOI = n+2		# n+2nd = EOI
	n += 3 			# n+3rd = start of new codes
	indexstream = vmycodify(r.flatten(), g.flatten(), b.flatten(), tbl=tbl) # turn each pixel into codes 0-255

	codes = [] # codes occuring after 255
	codestream = []
	indexbuffer = []
	
	##### Begining of LZW compression #####
	
	codestream.append(len(tbl))	# start with a clear code

	# get first index
	# codes and index buffer are lists
	indexbuffer.append(indexstream[0])
	#index stream
	for i in indexstream[1:]: #skip first one
		#get the next index
		K = i
		indexbuffer.append(K)
		if indexbuffer in codes:
			continue
		else:
			codes.append(indexbuffer)
			codestream.append(n)
			n+=1
			indexbuffer = [K]
			K = None
			#if largest code reached 4095
			if n == 4096: #send CC and clear tbl
				codestream.append(CC)
				n = EOI+1
				codes = []
	# end of loop
	codestream.append(n)
	codestream.append(EOI)
	##### End of LZW compression #####

	##### Write codes as bytes with flexible code sizes #####
	mybytes = bytearray()
	remainingbits = 0
	with open(outfile, 'ab') as out:
		# min LZW code size
		out.write(struct.pack('<B',8))
		currentsize = 9
		subblock = 1
		#subblock length
		mybytes.append(255)
		myblock = 1

		# GIF uses variable len bit encoding
		# python only lets me write in bytes
		# need to convert variable len bits to
		# bytes then write
		for i,n in enumerate(codestream, start=1):
			code = n[0]
			if remainingbits == 0:
				try: # trim down code to fit into 1 byte
					mybytes.append(code>>code.bit_length()-8)
					remainingbits = code>>8
				except ValueError: # code was smaller than a byte
					remainingbits = code>>8-code.bit_length()
				else:
					newcode = remainingbits<<8-remainingbits.bit_length()
					newcode += code>>8-remainingbits.bit_length()
				if newcode.bit_length() <= 8:
					mybytes.append(newcode)
					remainingbits = code>>8-remainingbits.bit_length()
				else:
					mybytes.append(newcode>>newcode.bit_length()-8)
					remainingbits = newcode>>8

			# when code grabbed is 2^n-1 
			# need to grab 1 more bit 
			if code == 2**currentsize-1:
				currentsize += 1
			elif code == 128: # clear code.
				#should appear when code size is 12
				currentsize = 9
    
			# when to start new subblock
			if subblock == 255:
				# introducing next subblock.
				# need to write remaining bits
				if remainingbits != 0:
					mybytes.append(remainingbits<<8-remainingbits.bit_length())
					remainingbits = 0
    
				if len(codestream)-i>=255:
					mybytes.append(255)
				else:
					mybytes.append(len(codestream)-i)
				subblock = 0
				myblock +=1
			subblock+=1
		# append block terminator
		mybytes.append(0x00)
		out.write(mybytes)
		out.close()
    
def parsecol(col, ch):
	'''Divide colour into its bucket. Just used midpoint of each colour bucket.'''
	buk = col
	if buk < 32 and ch != 'b':
		return 16
	elif buk < 64:
		return 48
	elif buk < 96 and ch != 'b':
		return 80
	elif buk < 128:
		return 112
	elif buk < 160 and ch != 'b':
		return 144
	elif buk < 192:
		return 176
	elif buk < 224 and ch != 'b':
		return 208
	else:
		return 240
parsecol = np.vectorize(parsecol)

#c2h = lambda r,g,b: 'x%02x%02x%02x' %(r,g,b)
def rgb2hex(pix):
	'''Convert r,g,b to hex values.'''
	#return c2h(pix[0],pix[1],pix[2])
	return np.array([hex(pix[0]), hex(pix[1]), hex(pix[2])])
vrgb2hex = np.vectorize(rgb2hex)

def makecolourtbl(myset):
	'''Make a colour table and append to file. GIF supports a maximum of 256 colours, but the video can have 256 per channel so we divide the channels into 8 buckets each.'''
	#tbl = np.zeros(768) 
	#enumerate over all the combinations using 3 bits for r, 3 bits for g, and 2 bits for b
	#for i in range(8):
	#	for j in range(8):
	#		for k in range(4):
	#                tbl[((i*32)+(j*4)+k)*3] = myset[i]
	#                tbl[((i*32)+(j*4)+k)*3+1] = myset[j]
	#                tbl[((i*32)+(j*4)+k)*3+2] = myset[k*2]
	tmp = np.array([16,48,80,112,144,176,208,240], dtype=np.uint8)
	n = tmp.shape[0]
	perm = np.empty((n,n,2), dtype=tmp.dtype)
	perm[..., 0] = tmp[:,None]
	perm[..., 1] = tmp
	perm = perm.reshape(-1,2)
	
	perm1 = np.hstack((perm, np.repeat(tmp[1], perm.shape[0]).reshape(-1,1)))
	perm2 = np.hstack((perm, np.repeat(tmp[3], perm.shape[0]).reshape(-1,1)))
	perm3 = np.hstack((perm, np.repeat(tmp[5], perm.shape[0]).reshape(-1,1)))
	perm4 = np.hstack((perm, np.repeat(tmp[7], perm.shape[0]).reshape(-1,1)))

	tbl = np.concatenate([perm1,perm2,perm3,perm4])
	tblb = np.apply_along_axis(rgb2hex, -1, tbl)
	#tblb = tbl.tobytes()
	#tblb = tbl.astype('<B')
	#vid = bytearray()
	#for b in tblb:
	#	vid.extend(b)
	#tbl.tofile('test.gif',format='%nn')
	#Application extension must appear immediately after the global colour table.
	ae = np.array([0x21, 0xFF, 0x0B, 0x4E, 0x45, 0x54, 0x53,
		0x43, 0x41, 0x50, 0x45, 0x32, 0x2E, 0x30,
		0x03, 0x01, 0x00, 0x00, 0x00], dtype=np.uint8)
	
	#tbl = np.reshape(tbl, (256,3))
	#for b in ae:
	#	vid.append(b)
     
	with open(outfile,'ab') as out:
		out.write(tblb.tobytes())
		out.write(ae.tobytes())
	return tbl

def writeGCE():
	'''Write Graphics Control Extension. Needed for animation.'''
	# Disposal method 2. We are throwing out the previous image and redrawing the whole thing.
	# Assuming video is taken at 32fps, a delay of about 4ms should be good.
	gce = np.array([0x21, 0xF9, 0x04, 0x08, 0x04, 0x00, 0x00, 0x00], dtype=np.uint8)
	#vid = bytearray()
	#for b in gce:
	#	vid.append(b)
	with open(outfile,'ab') as out:
		out.write(gce.tobytes())
def writeimagedescriptor(w,h):
	''' Write the image descriptor.'''
	#vid = bytearray()

	#Image Descriptor
	#image separator always begins with 2C
	#vid.append(0x2C)
	vid = np.array([0x2C,0x00,0x00,0x00,0x00],dtype=np.uint8)
	

	#next is left, top  position. Ignored by browsers now.
	#for i in range(4):
	#	vid.append(0x00)

	#image width and height little endian
	#packed as shorts(H) so only 2 bytes taken
	vid = np.append(vid, np.frombuffer(struct.pack('<H',w),dtype=np.uint8))
	vid = np.append(vid, np.frombuffer(struct.pack('<H',h),dtype=np.uint8))
	#vid.append(struct.pack('<H',w))
	#vid.append(struct.pack('<H',h))
	
	#packed field: Not using any options
	#vid.append(0x00)
	np.append(vid, np.array([0x00],dtype=np.uint8))
	
	with open(outfile,'ab') as out:
		out.write(vid.tobytes())
	
def writetitle(w,h):
	'''Write the header block.'''
	# create bytearray and append title
	#vid = bytearray()
	#for b in [0x47, 0x49, 0x46, 0x38, 0x39, 0x61]:
	#	vid.append(b)
	vid = np.array([0x47, 0x49, 0x46, 0x38, 0x39, 0x61], dtype=np.uint8)
	
	# append Logical Screen Descriptor
	# canvas width, height are ignored, set to 0
	#lsd = [0x00,0x00,0x00,0x00]
	vid = np.append(vid, np.array([0x00,0x00,0x00,0x00], dtype=np.uint8))
	vid = np.append(vid, np.frombuffer(struct.pack('<H',w), dtype=np.uint8))
	vid = np.append(vid, np.frombuffer(struct.pack('<H',h), dtype=np.uint8))
	#vid.extend(struct.pack('<H',w))
	#vid.extend()
	
	# using global colour table with bit-depth 8
	# unsorted, global colour table size 7
	#vid.append(0xF7)
	# background colour just use 0th colour.
	#vid.append(0x00)
	# pixel aspect ratio is ignored now
	#vid.append(0x00)
	vid = np.append(vid, np.array([0xF7,0x00,0x00], dtype=np.uint8))
	with open(outfile,'wb') as out:
		out.write(vid.tobytes())
	
readvid()
