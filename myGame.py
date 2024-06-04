from time import time

import cv2
import socketio

from myPose import myPose


class myGame:
    def __init__(self):
        self.pose = myPose()

        # Initialize a variable to store the state of the game (started or not).
        self.game_started = False

        # Initialize a variable to store the index of the current horizontal position of the person.
        # At Start the character is at center so the index is 1 and it can move left (value 0) and right (value 2).
        self.x_pos_index = 1

        # Initialize a variable to store the index of the current vertical posture of the person.
        # At Start the person is standing so the index is 1 and he can crouch (value 0) and jump (value 2).
        self.y_pos_index = 1

        # Initialize a counter to store count of the number of consecutive frames with person's hands joined.
        self.counter = 0

        # Initialize a variable to store the time of the previous frame.
        self.time1 = 0

        # Declate a variable to store the intial y-coordinate of the mid-point of both shoulders of the person.
        self.MID_Y = None

        # Initialize the number of consecutive frames on which we want to check if person hands joined before starting the game.
        self.num_of_frames = 5

        self.client = socketio.SimpleClient()
        self.client.connect("http://localhost:5000", transports=["websocket"])

    def move_JSD(self, JSD):
        # Check if the person has jumped.
        if JSD == "Jumping" and self.y_pos_index == 1:
            self.client.emit("character_movement", {"direction": "up"})

            # Update the veritcal position index of  the character.
            self.y_pos_index += 1

        # Check if the person has crouched.
        elif JSD == "Crouching" and self.y_pos_index == 1:
            self.client.emit("character_movement", {"direction": "down"})

            # Update the veritcal position index of the character.
            self.y_pos_index -= 1

        # Check if the person has stood.
        elif JSD == "Standing" and self.y_pos_index != 1:
            self.client.emit("character_movement", {"direction": "normal"})
            # Update the veritcal position index of the character.
            self.y_pos_index = 1

        return

    def play(self):
        # Initialize the VideoCapture object to read from the webcam.
        cap = cv2.VideoCapture(0)

        # Create named window for resizing purposes.
        cv2.namedWindow("Subway Surfers with Pose Detection", cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty(
        #     "Subway Surfers with Pose Detection",
        #     cv2.WND_PROP_FULLSCREEN,
        #     cv2.WINDOW_FULLSCREEN,
        # )
        while True:
            # Read a frame
            ret, image = cap.read()

            # Check if frame is not read properly then continue to the next iteration to read the next frame.
            if not ret:
                continue

            else:
                # Flip the frame horizontally for natural (selfie-view) visualization.
                image = cv2.flip(image, 1)

                # Get the height and width of the frame of the webcam video.
                image_height, image_width, _ = image.shape

                # Perform the pose detection on the frame.
                image, results = self.pose.detect_pose(
                    image, self.pose.pose_video, draw=self.game_started
                )

                # Check if the pose landmarks in the frame are detected.
                if results.pose_landmarks:
                    # Check if the game has started
                    if self.game_started:
                        # ------------------------------------------------------------------------------------------------------------------
                        # Commands to control the vertical movements of the character.
                        # ------------------------------------------------------------------------------------------------------------------
                        # Check if the intial y-coordinate of the mid-point of both shoulders of the person has a value.
                        if self.MID_Y:
                            image, JSD = self.pose.check_pose_JSD(
                                image, results, self.MID_Y, draw=True
                            )
                            self.move_JSD(JSD)

                    # Otherwise if the game has not started
                    else:
                        # Write the text representing the way to start the game on the frame.
                        cv2.putText(
                            image,
                            "JOIN BOTH HANDS TO START THE GAME.",
                            (5, image_height - 10),
                            cv2.FONT_HERSHEY_PLAIN,
                            2,
                            (0, 255, 0),
                            3,
                        )

                    # Command to Start or resume the game.
                    # ------------------------------------------------------------------------------------------------------------------
                    # Check if the left and right hands are joined.
                    if (
                        self.pose.check_hands_joined(image, results)[1]
                        == "Hands Joined"
                    ):
                        # Increment the count of consecutive frames with +ve condition.
                        self.counter += 1

                        # Check if the counter is equal to the required number of consecutive frames.
                        if self.counter == self.num_of_frames:
                            # Command to Start the game first time.
                            # ----------------------------------------------------------------------------------------------------------
                            # Check if the game has not started yet.
                            if not (self.game_started):
                                # Retreive the y-coordinate of the left shoulder landmark.
                                left_y = int(
                                    results.pose_landmarks.landmark[
                                        self.pose.mp_pose.PoseLandmark.RIGHT_SHOULDER
                                    ].y
                                    * image_height
                                )

                                # Retreive the y-coordinate of the right shoulder landmark.
                                right_y = int(
                                    results.pose_landmarks.landmark[
                                        self.pose.mp_pose.PoseLandmark.LEFT_SHOULDER
                                    ].y
                                    * image_height
                                )

                                # Calculate the intial y-coordinate of the mid-point of both shoulders of the person.
                                self.MID_Y = abs(right_y + left_y) // 2
                                # Update the value of the variable that stores the game state.
                                self.game_started = True

                                self.client.emit(
                                    "character_movement", {"direction": "space"}
                                )

                            # ----------------------------------------------------------------------------------------------------------
                            # Command to resume the game after death of the character.
                            # ----------------------------------------------------------------------------------------------------------

                            # Otherwise if the game has started.
                            else:

                                self.client.emit(
                                    "character_movement", {"direction": "space"}
                                )

                            # ----------------------------------------------------------------------------------------------------------
                            # Update the counter value to zero.
                            self.counter = 0

                    # Otherwise if the left and right hands are not joined.
                    else:
                        # Update the counter value to zero.
                        self.counter = 0

                # Otherwise if the pose landmarks in the frame are not detected.
                else:
                    self.counter = 0

                # Calculate the frames updates in one second
                # ----------------------------------------------------------------------------------------------------------------------
                # Set the time for this frame to the current time.
                time2 = time()

                # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
                if (time2 - self.time1) > 0:
                    # Calculate the number of frames per second.
                    frames_per_second = 1.0 / (time2 - self.time1)

                    # Write the calculated number of frames per second on the frame.
                    cv2.putText(
                        image,
                        "FPS: {}".format(int(frames_per_second)),
                        (10, 30),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (0, 255, 0),
                        3,
                    )

                # Update the previous frame time to this frame time.
                # As this frame will become previous frame in next iteration.
                self.time1 = time2

                cv2.imshow("Subway Surfers with Pose Detection", image)

                # Wait for 1ms. If a a key is pressed, check
                if cv2.waitKey(1) == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()


myGame = myGame()
myGame.play()
