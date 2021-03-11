import cv2
import csv
import numpy as np

if __name__ == "__main__":

    vid0 = cv2.VideoCapture('data/wibam/videos/NE_sync.mp4')
    vid1 = cv2.VideoCapture('data/wibam/videos/SE_sync.mp4')
    vid2 = cv2.VideoCapture('data/wibam/videos/SW_sync.mp4')
    vid3 = cv2.VideoCapture('data/wibam/videos/NW_sync.mp4')
    count = 0

    NS0_stat, NS2_stat, NS3_stat = 0,0,0
    EW1_stat, EW3_stat = 2,2

    csv_file_name = "light_states.csv"
    with open(csv_file_name, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        while True:
            ret0,frame0 = vid0.read()
            ret1,frame1 = vid1.read()
            ret2,frame2 = vid2.read()
            ret3,frame3 = vid3.read()

            # cv2.namedWindow('0', cv2.WINDOW_NORMAL)
            # cv2.namedWindow('1', cv2.WINDOW_NORMAL)
            # cv2.namedWindow('2', cv2.WINDOW_NORMAL)
            # cv2.namedWindow('3', cv2.WINDOW_NORMAL)
            # cv2.imshow('0', frame0)
            # cv2.imshow('1', frame1)
            # cv2.imshow('2', frame2)
            # cv2.imshow('3', frame3)
            # cv2.waitKey(0)

            if False in [ret0,ret1,ret2,ret3]:
                csv_file.close()
                break

            t0 = vid0.get(cv2.CAP_PROP_POS_MSEC)
            t1 = vid1.get(cv2.CAP_PROP_POS_MSEC)
            t2 = vid2.get(cv2.CAP_PROP_POS_MSEC)
            t3 = vid3.get(cv2.CAP_PROP_POS_MSEC)
            fps0 = vid0.get(cv2.CAP_PROP_FPS)
            fps1 = vid1.get(cv2.CAP_PROP_FPS)
            fps2 = vid2.get(cv2.CAP_PROP_FPS)
            fps3 = vid3.get(cv2.CAP_PROP_FPS)

            # print("{} {} {} {}".format(t0,t1,t2,t3), end="\r")

            state = ['r', 'y', 'g']
            states = [0,1,2]

            # if (np.argmax(NS0)==1 and NS0_stat==2) or \
            #    (np.argmax(NS0)==0 and NS0_stat==1) or \
            #    (np.argmax(NS0)==2 and NS0_stat==0):

            state_change = False
            
            NS0_r = np.sum(frame0[26,18])
            NS0_y = np.sum(frame0[41,27])
            NS0_g = np.sum(frame0[53,30])
            NS0 = np.array([NS0_r, NS0_y, NS0_g])
            if np.argmax(NS0) - NS0_stat ==-1 or np.argmax(NS0) - NS0_stat == 2:
                state_change = True
                NS0_stat = np.argmax(NS0)
                print("\nNS0 changed to {} at {}".format(state[NS0_stat], t0/1000))

            NS2_r = np.sum(frame2[65,204])
            NS2_y = np.sum(frame2[77,208])
            NS2_g = np.sum(frame2[87,210])
            NS2 = np.array([NS2_r, NS2_y, NS2_g])
            if np.argmax(NS2) != NS2_stat:
                state_change = True
                NS2_stat = np.argmax(NS2)
                print("\nNS2 changed to {} at {}".format(state[NS2_stat], t2/1000))
            
            NS3_r = np.sum(frame3[3,1577])
            NS3_y = np.sum(frame3[14,1573])
            NS3_g = np.sum(frame3[25,1570])
            NS3 = np.array([NS3_r, NS3_y, NS3_g])
            if np.argmax(NS3) != NS3_stat:
                state_change = True
                NS3_stat = np.argmax(NS3)
                print("\nNS3 changed to {} at {}".format(state[NS3_stat], t3/1000))


            EW1_r = np.sum(frame1[59,65])
            EW1_y = np.sum(frame1[69,66])
            EW1_g = np.sum(frame1[82,70])
            EW1 = np.array([EW1_r, EW1_y, EW1_g])
            if np.argmax(EW1) != EW1_stat:
                state_change = True
                EW1_stat = np.argmax(EW1)
                print("\nEW1 changed to {} at {}".format(state[EW1_stat], t3/1000))

            EW3_r = np.sum(frame3[7,156])
            EW3_y = np.sum(frame3[19,160])
            EW3_g = np.sum(frame3[34,166])
            EW3 = np.array([EW3_r, EW3_y, EW3_g])
            if np.argmax(EW3) != EW3_stat:
                state_change = True
                EW3_stat = np.argmax(EW3)
                print("\nEW3 changed to {} at {}".format(state[EW3_stat], t3/1000))

            # print("NS0 {}, NS2 {}, NS3 {}, EW1 {}, EW3 {}".format(
            #     state[NS0_stat], state[NS2_stat], state[NS3_stat],
            #     state[EW1_stat], state[EW3_stat]
            # ), end="\r")

            print("Time: {:.2f}".format(t0/1000), end='\r')

            if state_change:
                csv_writer.writerow([str(t0/1000), str(NS0_stat), str(NS2_stat), str(NS3_stat), str(EW1_stat), str(EW3_stat)])

            

            count = count + 1
            if cv2.waitKey(10) & 0xFF == ord('q'):
                csv_file.close()
                break
