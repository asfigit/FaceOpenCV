import cv2 as cv

#Working with Images

# img = cv.imread('Resources/Photos/park.jpg')
# cv.imshow('Image',img)

# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('Gray',gray)


# cv.waitKey(0)

# Working with Video

vid = cv.VideoCapture('Resources/Videos/kitten.mp4')

while True:
    isTrue, frame = vid.read()
    
    if isTrue:
        cv.imshow('Video',frame)
        
        if cv.waitKey(20) & 0xFF == ord('d'):
            break
    else:
        break

vid.release()
cv.destroyAllWindows()
