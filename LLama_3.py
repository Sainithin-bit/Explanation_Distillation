import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForCausalLM
import torch.nn.functional as F



data= '''
4: #C A motorcycle rides past C.
5: #C C drives the car with the steering wheel in his hands
6: #C C drives the car with the steering wheel in his hands
7: #C C drives past a motorcycle.
8: #C C drives past a motorcycle.
9: #C A motorcycle rides past C.
89a3217e-f46a-4a96-a02a-db5218a16435.mp4
0: #C C looks around
1: #C C turns left on the road
2: #C C turns the car steering with his right hand.
3: #C A car drives past C.
4: #C A motorcycle rides past C.
5: #C A motorcycle rides past C.
6: #C A white car drives past C.
7: #C C drives the car on the road with both hands.
8: #C A motorcycle rides past C.
9: #C C drives the car with his right hand on the steering wheel
9b16f8ed-efc7-4f89-bbbe-9be9dc9186c7.mp4
0: #C C drives the car
1: #C C turns right 
2: #O The man A operates the phone with both hands.
3: #C C holds steering wheel
4: #C C drives the car
5: #C C drives the car with his right hand.
6: #C C turns left
7: #C C drives the car with the steering wheel in his left hand
8: #C C drives the car with the steering wheel in his right hand
9: #O The man X operates the steering wheel with his right hand.
e281b473-76d3-46ac-892d-adb8f6e1d5f0.mp4
0: #C C drives past a car.
1: #C C drives past a motorcycle.
2: #C C drives past a car.
3: #O The man B drives past a motorcycle.
4: #O The man B operates the phone with his right hand.
5: #C C drives the car with the steering wheel in both hands
6: #O A woman X operates a phone with her right hand.
7: #C C turns the steering wheel with both hands.
8: #O The man B operates the phone with his right hand.
9: #C A motorcycle rides past C.
6ed6caf3-c660-4f1f-8c1f-d2a4d4e6cfce.mp4
0: #C C turns the car steering with his right hand.
1: #C C moves the car gear
2: #C C turns the steering wheel with both hands.
3: #C C drives the car with his right hand on a steering wheel
4: #C C stops the car 
5: #C C drives the car with the steering wheel in his hands.
6: #C C drives car on the road
7: #C C drives the car
8: #C C drives a car on the road
9: #C A white car drives past C.
0a4557bf-9b3e-4d73-861a-04b78b5a9306.mp4
0: #C C stops car on traffic
1: #C C drives car on the road
2: #C C drives the car
3: #C C drives on the road
4: #C C looks around
5: #C C drives car on the road
6: #C C drives car on the road
7: #C C turns left on the road.
8: #C C turns the car left
9: #C C drives the car
05413f3e-662e-487c-9817-1a9d2b58971d.mp4
0: #C A black car drives past C
1: #C A car drives past C.
2: #C A car drives past C.
3: #C A car drives past C.
4: #C A car drives past C.
5: #C A white car drives past C
6: #C A car drives past C.
7: #C A blue car drives past C
8: #C A car drives past C.
9: #C A car drives past C.
3997a580-77d2-406e-a16c-872026d88a91.mp4
0: #C C drives a car 
1: #C C drives the car with his right hand on the steering wheel.
2: #C A motorcycle rides past C.
3: #C A car drives past C.
4: #C C drives the car with his right hand on the steering wheel.
5: #C C drives past the silver car.
6: #O The man A operates the phone in his right hand with his left hand
7: #C C drives past a car.
8: #C A black car drives past C.
9: #O A man Y operates a phone with both hands.
8ddd487e-1217-4ffa-b9a4-71ac557d3ecb.mp4
0: #C A black car drives past C.
1: #C C drives the car with his right hand.
2: #C C drives the car on the road with his right hand on the steering wheel
3: #O The man B holds a phone with his right hand.
4: #C A motorcycle rides past C.
5: #O A red car drives past C.
6: #O A motorcycle rides past C.
7: #O A motorcycle rides past C.
8: #C A man A rides a bicycle past C.
9: #O A man Z rides a motorcycle past C.
f444c2af-f997-4b92-8240-2846591d6765.mp4
0: #C C stops the car at the traffic light.
1: #C A black car drives past C.
2: #C A white car drives past C.
3: #C A white car drives past C.
4: #C A white car drives past C.
5: #C A car drives past C.
6: #C A motorcycle rides past C.
7: #C A white car drives past C.
8: #C A white car drives past C.
9: #C A white car drives past C.
5a1afad4-6542-49db-8107-f8dda83823c7.mp4
0: #C A white car drives past C
1: #C A white car drives past C
2: #C A white car drives past C
3: #C A car drives past C.
4: #C A white car drives past C
5: #C A grey car drives past C
6: #C A white car drives past C
7: #C C drives past a white car
8: #C C drives past a car.
9: #C C drives past a car.
8daa8a74-3740-4a8c-8ad3-0b5f766ed35b.mp4
0: #C C looks around the car
1: #O The man B operates the phone with both hands.
2: #O The man D operates a phone with his left hand.
3: #O The man X operates the phone with his right hand.
4: #C A car drives past C.
5: #O a person M uses a phone
6: #C C drives the car with the steering wheel in his right hand
7: #C C looks around the car
8: #C C drives car on the road
9: #C C looks around
dd10a29f-e876-4290-801b-bd2fd6e464ba.mp4
0: #O The man B holds the phone with his right hand.
1: #O A man X uses a phone 
2: #C C looks around
3: #C C looks around
4: #O Person A uses a phone
5: #C A yellow car drives past C.
6: #C C drives car on the road
7: #C A car drives past C.
8: #O A man P operates a phone with both hands.
9: #O A man X uses the phone 
71e6651c-5588-49a5-84f0-6fc67ef24c60.mp4
0: #O The man X operates a phone with his right hand
1: #C A black car drives past C.
2: #O A man X operates a phone with his right hand
3: #C A motorcycle rides past C.
4: #C A black car drives past C.
5: #C A motorcycle rides past C.
6: #O A car drives past C.
7: #C A white car drives past C.
8: #C A motorcycle rides past C.
9: #O A man D operates a phone with his right hand
aacb00ca-5063-489c-b0e0-2c7fc280e027.mp4
0: #C C drives past a man G.
1: #C C drives past a motorcycle.
2: #C C turns the steering wheel with his right hand.
3: #C C drives past a man G.
4: #C C turns the steering wheel with his right hand.
5: #C C drives the car on the road with both hands on the steering wheel.
6: #C C drives the car
7: #C C drives past a bus.
8: #C C turns left on the road.
9: #C C drives the car with his left hand on the steering wheel.
443b575e-e7cb-41ca-8aee-cee2d0ff8d9d.mp4
0: #C C looks around
1: #C C turns left 
2: #O A man X drives a car
3: #C C looks at the road
4: #C C drives the car
5: #O A man X uses phone
6: #O A man Y touches his face with his right hand
7: #O Person A holds the phone
8: #C C turns to the side
9: #O person B turns the head
219b7f45-f6da-438a-85a4-034641b312fc.mp4
0: #C A black car drives past C.
1: #C A motorcycle rides past C.
2: #O The man A holds a phone with both hands.
3: #C C drives the car with the steering wheel in his right hand.
4: #O A man D drives the car on the road.
5: #O A yellow car drives past C.
6: #C C drives the car with his right hand.
7: #O The man A operates the phone with his right hand.
8: #C C drives the car with the steering wheel in his left hand.
9: #C A yellow car drives past C.
d944d5e6-b737-49c2-9361-7d26881dda17.mp4
0: #C A blue car drives past C.
1: #O The man Y holds a phone with his left hand
2: #O The man Y presses the phone with his left hand
3: #C C drives past a car.
4: #C A white car drives past C.
5: #O The man X operates the phone in his right hand
6: #C A white car drives past C.
7: #C A black car drives past C.
8: #C A white car drives past C.
9: #C A red car drives past C.
711bf2ae-5ff2-4751-9bc9-0ae37a9797e0.mp4
0: #C C drives past a motorcycle.
1: #O The woman X operates the phone with her right hand.
2: #C A motorcycle rides past C.
3: #C C drives past a motorcycle.
4: #C A yellow car drives past C.
5: #C C drives past a yellow car.
6: #C A motorcycle rides past C.
7: #C C drives past a bike.
8: #O The man B operates the phone with both hands.
9: #C A yellow car drives past C.
5da436ad-2e39-4a7c-9349-740e53704444.mp4
0: #C C drives past a bike on the road
1: #C C drives past a motorcycle.
2: #C C drives past a motorcycle.
3: #C C drives past a bike.
4: #C C drives past a motorcycle.
5: #C C drives past a motorcycle.
6: #C A car drives past C.
7: #C C drives past a motorcycle.
8: #C A motorcycle rides past C.
9: #C C drives past a bike.
3bf7bbc3-5f92-4413-97fe-322545ec5574.mp4
0: #C C Drives a car on the road
1: #C A car drives past C.
2: #O A man Y operates a phone
3: #C A black car drives past C.
4: #C A car drives past C.
5: #O The man A holds the phone with his right hand
6: #C A black car drives past C.
7: #C C turns left on the road.
8: #C A car drives past C.
9: #C A white car drives past C.
b5532d42-f779-4432-befd-409ec3b69b61.mp4
0: #O The man A operates the phone with his left hand
1: #C A car drives past C.
2: #C A black car drives past C.
3: #O The man A operates a phone with both hands.
4: #O A man B operates a phone with his right hand
5: #O A man B operates a phone with his left hand
6: #O A man X operates a phone with his left hand
7: #O A man D operates a phone with his left hand
8: #C A white car drives past C.
9: #O A man X operates a phone with his right hand
0c4cc445-c20a-423e-8f7f-b6cfe471d085.mp4
0: #O The man X drives the car with his right hand.
1: #O The man X touches his head with his left hand.
2: #O The man X touches his head with his left hand.
3: #O The man X gesticulates with his right hand.
4: #O A white car drives past C.
5: #O A white car drives past C.
6: #O The man X lowers his left hand.
7: #O The man X adjusts a visor on his face with his left hand
8: #O A white car drives past C.
9: #O The man X adjusts the car sun visor with his left hand.
2f7681ea-fa61-48c3-8f92-9871d0de1e69.mp4
0: #C A motorcycle rides past C.
1: #C A black car drives past C.
2: #C A motorcycle rides past C.
3: #C A car drives past C.
4: #O A man X drives the car on the road with both hands on the steering wheel
5: #C A car drives past C.
6: #O The man A drives past a motorcycle.
7: #C A car drives past C.
8: #C A car drives past C.
9: #C A white car drives past C.
3b7d602d-843d-4ad0-8150-a46a412f3b3c.mp4
0: #C A motorcycle rides past C.
1: #O A man E rides a motorcycle with both hands on the handlebar
2: #C A motorcycle rides past C.
3: #C A motorcycle rides past C.
4: #C A motorcycle rides past C.
5: #O A white car drives past the car C is in.
6: #C C drives the car
7: #O A man D holds a phone with his left hand.
8: #O A man J holds a phone with his right hand.
9: #C A white car drives past C.
0fbc99d9-4987-4db8-9353-546a296edb8d.mp4
0: #C A car drives past C.
1: #C A car drives past C.
2: #C A car drives past C.
3: #C A car drives past C.
4: #C A motorcycle rides past C.
5: #C A motorcycle rides past C.
6: #C A motorcycle rides past C.
7: #O A man Y operates a phone in his right hand
8: #C A motorcycle rides past C.
9: #C A black car drives past C.
a22cae98-eb21-4b2e-b853-b2a3d617fbf7.mp4
0: #C C drives the car on the road with the steering wheel in his right hand.
1: #C C turns right
2: #O A man Y drives the car 
3: #O A man X drives a car
4: #C C looks at the road
5: #C C looks at the window
6: #C C drives the car on the road with his right hand on the steering wheel.
7: #C C looks around
8: #C C drives a car on the road
9: #C C drives the car with both hands on the steering wheel.
db071bb8-63dd-48ad-bf86-0249955bdb42.mp4
0: #C A black car drives past C.
1: #C A motorcycle rides past C.
2: #C C drives the car with his right hand.
3: #C A black car drives past C.
4: #C A car drives past C.
5: #C A car drives past C.
6: #C A car drives past C.
7: #C A car drives past C.
8: #C A car drives past C.
9: #C A car drives past C.
cd6ad248-95ae-4f77-96c7-64e1c1a4838f.mp4
0: #C C turns the steering wheel to the
1: #C C holds the steering wheel with both
2: #C C turns to the right 
3: #O Person A holds a phone
4: #C C turns left 
5: #C C turns the steering wheel
6: #C C looks on the side mirror
7: #C C looks around
8: #C C turns right
9: #C C turns left 
56fc1498-8a1b-4eaa-85c6-9bb0d527fee5.mp4
0: #C C drives the car with his right hand on the steering wheel
1: #C C drives the car
2: #C C drives past a motorcycle.
3: #C C drives car on the road
4: #C C looks at the road
5: #C C drives the car with both hands.
6: #C C looks at the side mirror
7: #C C drives past a bike on the road.
8: #C C drives the car with his right hand.
9: #C C drives the car with his left hand.
4b76d043-475c-4a95-adae-cff7e48af7a3.mp4
0: #C C drives past a white car.
1: #O A white car drives past C.
2: #O A white car drives past C.
3: #O The man X drives the car on the road with the steering wheel in his right hand
4: #C A blue car drives past C.
5: #C C drives the car with his right hand.
6: #C C drives the car
7: #O A yellow car drives past C.
8: #O The man B adjusts the gear of the car with his right hand.
9: #O a person P drives the car
943d3b2c-0b21-4b6a-a54a-9de0ae098c88.mp4
0: #C C holds steering wheel with left hand
1: #C C holds steering wheel 
2: #C C places his right hand on his lap.
3: #O The man X places his left hand on his thigh.
4: #C C turns right
5: #C C looks around
6: #O The man B places his right hand on his right thigh.
7: #O The man X operates the car gear with his right hand.
8: #C A yellow car drives past C.
9: #C C drives the car on the road with his right hand.
360e58bd-7a0d-4c78-8ad7-b82e87e2ad1e.mp4
0: #C C drives past a blue car.
1: #C C drives past a motorcycle.
2: #C C holds the gear shift with his left hand
3: #C C moves the gearshift with his left hand
4: #C C drives past a motorcycle.
5: #C C turns the steering wheel with his right hand
6: #C C drives past a car.
7: #C C drives past a motorcycle.
8: #C C drives past a car. 
9: #C C drives past a bike.
d21bb6ee-f994-4adf-8742-e63cc017f3a3.mp4
0: #C A motorcycle rides past C.
1: #C C drives past the motorcycle.
2: #C A motorcycle rides past C.
3: #C C drives past a motorcycle.
4: #C A motorcycle rides past C.
5: #C C drives past a motorcycle.
6: #C C drives past a bike.
7: #C A motorcycle rides past C.
8: #C A motorcycle rides past C.
9: #C C drives the car with the steering wheel in both hands
1947c7ff-6fc6-4444-9ee7-c2ed241a5db6.mp4
0: #C C looks around the road.
1: #C C Moves a hand 
2: #C C drives the car with his left hand on the steering wheel
3: #C C looks at the side mirror
4: #C C Looks around a parking lot
5: #C C turns to the left.
6: #O A man X holds a phone
7: #C C turns the car steering with his right hand.
8: #C C drives a car 
9: #C C looks at a car 
b89972f2-84a9-402b-be35-c00bbe42f583.mp4
0: #C C turns the steering wheel to the left 
1: #C C drives a car
2: #C C looks at the road
3: #C C drives the car with the right hand
4: #C C drives a car 
5: #C C looks at the windscreen
6: #C C holds the steering wheel with his right hand
7: #O Person A operates the phone
8: #C C looks at the windscreen
9: #C C drives a car
3aacfbc2-d610-4d6a-95f3-55418f4d691f.mp4
0: #C C drives the car with both hands.
1: #C A white car drives past C.
2: #C A car drives past C.
3: #C A white car drives past C.
4: #C A car drives past C.
5: #C C drives past a car.
6: #C C pushes the gearshift with his left hand.
7: #C C drives past a white car.
8: #C C drives past a car.
9: #O The man X turns the car steering wheel with his right hand
e1f28274-4092-49d8-a851-4c181757da8a.mp4
0: #O The man X operates the gear of the car with his right hand.
1: #O A man H drives the car with both hands.
2: #C C drives past a white bus.
3: #C C drives past a white car.
4: #C C drives the car with his left hand on the steering wheel.
5: #O The man B holds the phone with his right hand.
6: #C C looks around
7: #C A motorcycle rides past C.
8: #C A motorcycle rides past C.
9: #C A car drives past C.
9e7b6daa-f2dd-4e8a-a9ca-e1a039e82971.mp4
0: #O A man X uses phone 
1: #C C looks around
2: #O A man Z operates a phone with his left hand.
3: #C C looks at the road
4: #C C looks around 
5: #O A man A operates a phone with his right hand.
6: #O The man X holds the car steering wheel with his right hand
7: #O A man X operates a phone with both hands.
8: #O The man Y turns the steering wheel with his right hand.
9: #C C looks around
28439e4d-079f-4e13-83a0-5549f3c86db0.mp4
0: #C C drives the car with his right hand.
1: #C A white car drives past C.
2: #C C drives the car with both hands on the sterring wheel.
3: #C C taps the steering wheel with his right hand.
4: #C C drives past a car.
5: #C C taps the steering wheel with his right hand.
6: #C A black car drives past C.
7: #C C places his left hand on his right hand.
8: #C C raises his left hand.
9: #C C drives the car with the steering wheel in his right hand.
1350204a-f6a1-4602-8b4e-55271edfe6c1.mp4
0: #O The man A operates a phone in his left hand
1: #O person X drives car
2: #O The man X operates the phone with his right hand
3: #C C drives car on the road
4: #C C turns right
5: #O A man Z operates a phone with his left hand
6: #O Person A holds the phone
7: #C C looks around
8: #O A car drives past C.
9: #C C looks around
5a73980a-6c9f-48d9-b3fd-cb7c62881ad6.mp4
0: #C A car drives past C.
1: #C C holds the steering wheel with his left hand.
2: #C C drives past a car.
3: #C C drives the car with his right hand.
4: #C C drives past a white car.
5: #C C drives the car with his left hand.
6: #C A black car drives past C.
7: #C C drives past a bike on the road.
8: #C C drives the car with his left hand on the steering wheel
9: #C A white car drives past C.
b765b96c-1474-4f9d-9614-46ccd532d05a.mp4
0: #O The man B holds the phone with both hands.
1: #C C looks around
2: #O A man D operates a phone with both hands.
3: #O A man E operates a phone in a car with his right hand.
4: #O A man B operates a phone with both hands.
5: #C C looks around
6: #C A motorcycle rides past C.
7: #O The man A operates the phone with his right hand.
8: #O The man A holds a phone with his left hand.
9: #C A silver car drives past C.
68026838-9631-4a10-ae66-2da5e4132bdb.mp4
0: #C C drives car on the road
1: #O person B operates a phone
2: #O A man X operates a phone
3: #O Person B operates a phone
4: #C C drives the car with the steering wheel in his left hand
5: #O The man A operates the phone with his right hand.
6: #O The man B holds the phone with his left hand.
7: #C C turns left on the road
8: #C C looks at the road
9: #O person A scrolls a phone
b38f5bcc-c5e4-4d7f-ad76-0dec071506d0.mp4
0: #C C drives the car with both hands.
1: #C C drives a car on the road 
2: #C C drives the car with his right hand on the steering wheel.
3: #C C drives the car with the steering wheel in both hands.
4: #C C holds the steering wheel with his right hand.
5: #C C drives the car with his right hand.
6: #C C drives the car with his right hand.
7: #C C turns the steering wheel with his right hand.
8: #C C drives the car with both hands. 
9: #C C drives the car with both hands.
9e744d07-4741-4fe4-b2be-21666fed413b.mp4
0: #C C drives the car with his right hand on the steering wheel.
1: #C C drives the car with the steering wheel in  both hands.
2: #O A man X operates a phone with his left hand.
3: #C C drives past a black car.
4: #C C removes his right hand from the steering wheel of the car.
5: #C C drives the car with his right hand.
6: #C C drives the car with his left hand on the steering wheel.
7: #C C drives the car on the road with his right hand.
8: #C C drives the car with the steering wheel in his left hand.
9: #C C drives the car with his right hand.
af162e78-1d80-486f-92c4-2641b0213179.mp4
0: #C C turns right
1: #C C drives the car
2: #C C holds the steering wheel with both hands
3: #C C turns right
4: #C C holds the steering with his left hand.
5: #C C turns the steering wheel with his right hand.
6: #C C drives the car 
7: #C C drives the car with his left hand.
8: #C C drives the car with the steering wheel in his right hand.
9: #C C drives the car 
61061fc9-07b6-4db1-ae6c-0686546ec554.mp4
0: #C C drives a car
1: #C C turns right
2: #C C turns the car steering with his right hand.
3: #C C drives a car
4: #C A motorcycle rides past C.
5: #O A woman Q operates a phone with both hands.
6: #C C drives the car with the steering wheel in his right hand
7: #C C drives past a motorcycle.
8: #C C drives the car with the steering wheel in his right hand
9: #C C turns the car steering with his right hand.
6782229f-f752-4a91-9a48-546fd791bd48.mp4
0: #O The man A operates the phone with his left hand.
1: #O A woman Y moves hand on the steering wheel
2: #O a person X drives a car
3: #C C turns right
4: #O The man X operates the phone with his right hand.
5: #C C drives the car with the steering wheel in both hands
6: #C C drives the car 
7: #O A woman X drives the car
8: #C C turns right
9: #C C turns to the side
735bd135-f306-4ae6-b75b-c1894dbc7a47.mp4
0: #C C looks around
1: #C C turns on the car
2: #C C turns the car left.
3: #C C drives the car with the steering wheel in his right hand
4: #C C drives the car with his left hand.
5: #C C drives past a bike on the road.
6: #C C drives the car with the steering wheel in his right hand
7: #C C drives the car with his left hand on the steering wheel
8: #C C drives the car with the car steering in his right hand
9: #C C turns on the headlights of the car with his left hand
1440e4e0-4ad0-49ff-a9f3-8d38a491b6c1.mp4
0: #C C drives car on the road
1: #C C turns left on the road
2: #C C drives the car 
3: #C C drives the car
4: #C C turns to the side
5: #C C turns the steering wheel with his right hand.
6: #C C turns right
7: #C C turns the steering wheel
8: #C C turns left on the road.
9: #C C drives the car
0db5a953-150e-455a-9ca1-8e77633b9e2a.mp4
0: #O The man B operates the phone with his right hand.
1: #C C looks around
2: #C C gestures with right hand.
3: #C C looks around
4: #O A man Z sits on a car seat.
5: #C C drives a car.
6: #O A man Y holds a phone in his left hand.
7: #C C looks around
8: #C C drives the car with her right hand.
9: #C C looks at the road
c6216c42-197b-4b18-952f-86db4516c653.mp4
0: #O The woman X turns the steering wheel with both hands.
1: #C C drives the car with his right hand.
2: #C C drives the car
3: #O The woman D holds the car gear with her right hand.
4: #O A man J holds a phone with his right hand.
5: #C C drives the car
6: #O The woman B holds the car steering with her right hand.
7: #C C drives the car with the car steering in his right hand.
8: #C C drives past a man K.
9: #C A motorcycle rides past C.
db720ae7-75ec-4fcd-ac98-7ec6c6afdf7c.mp4
0: #C C drives the car with both hands on the steering wheel
1: #C C drives past a truck.
2: #C C drives past a car.
3: #C C turns the car steering with his right hand.
4: #C C drives past a white car.
5: #C A black car drives past C.
6: #O The woman A holds a phone with her left hand.
7: #C A yellow car drives past C.
8: #C A car drives past C.
9: #C A black car drives past C.
4ed616ef-279d-4936-b0d1-c1c4336ad25a.mp4
0: #C A motorcycle rides past C.
1: #C C drives past a bike. 
2: #C A motorcycle rides past C.
3: #C A motorcycle rides past C.
4: #C A white car drives past C.
5: #C C drives past a bike on the road
6: #C A white car drives past C.
7: #C C drives past a motorcycle.
8: #C C drives past the blue car.
9: #C A black car drives past C.
a45cc582-c639-42f0-823e-2e1a16515d36.mp4
0: #C A car drives past C.
1: #C A white car drives past C.
2: #C A white car drives past C.
3: #C A white car drives past C.
4: #C A white car drives past C.
5: #C A white car drives past C.
6: #C A car drives past C.
7: #C A car drives past C.
8: #C A white car drives past C.
9: #C A white car drives past C.
3c9ff6c7-c5fe-48b8-a92f-cd3f15ff8fe4.mp4
0: #C C drives car on the road
1: #C C turns the steering wheel with his right hand.
2: #C C holds the car gear with his right hand.
3: #C C drives the car 
4: #C C drives car on the road
5: #C C drives car on the road
6: #C C drives car
7: #C C looks around
8: #C C drives car on the road
9: #C C moves car steering
124a66eb-e6ca-4d39-8754-f53ecbf2b72a.mp4
0: #C C drives past the motorcycle.
1: #C A motorcycle rides past C.
2: #C C drives past a bike on the road
3: #C A motorcycle rides past C.
4: #C A motorcycle rides past C.
5: #C A motorcycle rides past C.
6: #C A motorcycle rides past C.
7: #C A motorcycle rides past C.
8: #C A motorcycle rides past C.
9: #C a motorcycle drives past C.
7ced5519-163c-4e52-8d92-c80cfba38247.mp4
0: #C C drives the car with both hands on the steering wheel.
1: #O A woman F holds a car steering with her right hand.
2: #O The woman X places her right hand on the steering wheel.
3: #C C drives the car with both hands on the steering wheel.
4: #O The woman X touches the steering wheel with her right hand.
5: #O The woman A moves the car gear with her right hand.
6: #O The woman X turns the car steering with her right hand.
7: #C C drives the car
8: #O The woman A passes the phone from her left hand to her right hand
9: #O The woman X places her right hand on her right leg.
dee08378-88f2-4571-9ddc-17dd665a78ee.mp4
0: #C C holds the steering wheel with his left hand.
1: #C C looks around the car
2: #C C drives the car on the road with his right hand
3: #C C looks at the road
4: #C C holds a steering wheel with his right hand.
5: #C C looks around
6: #C C drives the car
7: #C C looks at the road
8: #C C looks at the road
9: #C C drives the car.
fb126ed5-ac4b-43a6-a916-6a2a94481e53.mp4
0: #O The man X adjusts the gear with his right hand.
1: #O The man A operates the phone in his right hand with his right hand
2: #O A white car drives past C.
3: #O The man A drives the car with his right hand on the steering wheel
4: #C C drives the car with both hands on the steering wheel.
5: #O The man X presses the phone with his right hand.
6: #O A yellow car drives past C.
7: #C A man K rides a bicycle past C.
8: #O The man X holds the phone with his right hand.
9: #O The man X holds the car gear shift with his right hand.
853019c3-858e-4e03-bb03-30e2305c51ae.mp4
0: #C C drives the car with the steering wheel in his hands. 
1: #O The man A drives past a red car.
2: #O A yellow car drives past C.
3: #O The man A holds a phone with both hands.
4: #O man X drives car
5: #O The man A holds the phone with both hands.
6: #O The man A operates a phone with both hands.
7: #C C looks around
8: #O The woman B holds the phone with both hands.
9: #C C drives past the yellow car.
2efb6573-607f-4348-9eca-54052ea09d20.mp4
0: #C C drives the car with the car steering in his left hand
1: #C C drives past a red car.
2: #C C drives the car with the steering wheel in his left hand
3: #C A motorcycle rides past C.
4: #C C drives past a white car.
5: #C C holds the steering wheel with his left hand.
6: #C C drives past a white car.
7: #C C drives past a car.
8: #C C drives past a white truck.
9: #C C drives past a red car.
776452c4-09d2-4ff2-887d-213abe4a05a7.mp4
0: #C C drives past a truck.
1: #C C drives past a truck.
2: #C C interacts with the man X.
3: #C C drives past a truck.
4: #C C drives past a car.
5: #C C points at a car on the road with his right index finger
6: #C A white bus drives past C.
7: #C C drives the car with the steering wheel in his right hand.
8: #C C drives past a white truck.
9: #C C raises his right hand.
78e42844-e8fa-4543-8bce-bb4c987825eb.mp4
0: #O A man X drives the car 
1: #O The man A operates a phone with both hands.
2: #O Person A operates a phone
3: #O person B holds the phone
4: #O person B operates a phone
5: #C C drives a car 
6: #O A man X operates a phone
7: #C C looks around
8: #C C looks around
9: #O A man X uses phone
a3de0ffe-a439-411c-a12f-04bdcc193071.mp4
0: #C C drives past a car.
1: #C C drives past a car.
2: #C C drives past a motorcycle.
3: #O The man B holds the phone with his left hand.
4: #C C drives past a car.
5: #O A black car drives past C.
6: #O The woman A operates the phone with her right hand.
7: #C C drives past a car. 
8: #C C drives past a car.
9: #C C drives past a motorcycle.
fb63cd92-90de-496c-97ad-99c7bfbd8cba.mp4
0: #C A motorcycle rides past C.
1: #C A motorcycle drives past C.
2: #C A motorcycle rides past C.
3: #C A motorcycle rides past C.
4: #C C drives past a motorcycle.
5: #C C drives past a black car.
6: #C C drives past a black car.
7: #C A red car drives past C.
8: #C C drives past a motorcycle.
9: #C C drives past a motorcycle.
e433bb68-9635-4eb1-994f-6c488e58e137.mp4
0: #C C presses a button in the car with his right hand.
1: #C C holds the car key
2: #C C holds the car key with his left hand.
3: #C C drives the car with both hands on the sterring wheel.
4: #C C drives past a truck.
5: #C C drives the car on the road with both hands on the steering wheel.
6: #C C drives the car with the steering wheel in both hands.
7: #C C turns the car on
8: #C C drives past a white car.
9: #C C drives past a truck.
193ad9a1-5ffe-4dbd-9545-5192e6dc200a.mp4
0: #C A blue car drives past C.
1: #C C drives past a white car.
2: #C A white car drives past C.
3: #C A car drives past C.
4: #C C looks around
5: #O The man B removes his right hand from the car steering
6: #O A man P operates a phone with his right hand.
7: #C C drives past a bus.
8: #C A motorcycle rides past C.
9: #C A motorcycle rides past C.
96aa4693-228a-4e91-9d0b-c4b05f5237d3.mp4
0: #O A man A operates a phone with both hands.
1: #O A man P operates a phone with his right hand
2: #C A white car drives past C.
3: #C A car drives past C.
4: #O A man Z operates a phone with his right hand
5: #C C looks around
6: #O A man Y operates a phone with both hands.
7: #O A woman P operates a phone with her right hand
8: #O The woman B operates the phone with her right hand
9: #C A motorcycle rides past C.
7db4fcf9-5ef7-4a1d-a9bf-dcb2d9842ffe.mp4
0: #O A man Y drives a car on the road.
1: #O The man A holds the steering wheel with his right hand.
2: #C C turns the steering wheel with his left hand.
3: #C C drives the car on the road with his right hand on the steering wheel
4: #C C drives the car with both hands.
5: #C C turns to the side
6: #O The man X operates the gear shift with his right hand.
7: #C C drives the car with his right hand.
8: #C C drives the car with his right hand.
9: #O The man B operates the phone with both hands.
216fe158-8f62-4068-b313-1480e07795b9.mp4
0: #O A man X drives the car
1: #C C looks around
2: #C C looks around
3: #O The man A holds the phone with his right hand.
4: #O A man Y drives a car 
5: #O The man A holds the phone with his left hand.
6: #O A man Z drives the car 
7: #O A car drives past C.
8: #C C drives the car
9: #O A man X drives the car
80fc8eba-c9e6-43da-9c5b-c0382e327615.mp4
0: #C C drives past a car
1: #C A car drives past C
2: #C C drives a car
3: #C A car drives past C
4: #C A car drives past C
5: #C A car drives past C
6: #C A car drives past C
7: #C A car drives past C
8: #C A car drives past C
9: #C A car drives past C
5bcc0f0d-5831-46d5-b7ae-5f689cf1ea24.mp4
0: #C C drives past a white car.
1: #C C drives the car on the road with both hands.
2: #C C removes his right hand from the steering wheel.
3: #C C turns the steering wheel with his right hand.
4: #C A car drives past C.
5: #C C stops the car
6: #C A car drives past C.
7: #C A white car drives past C.
8: #C C drives the car with both hands on the steering wheel.
9: #C A car drives past C.
240890bc-8241-4893-af8c-2dc9cc1346cd.mp4
0: #O A man A operates a phone with his left hand.
1: #C A car drives past C.
2: #C C drives the car with the car steering in both hands.
3: #C C drives the car with both hands on the steering wheel.
4: #C A car drives past C.
5: #C A black car drives past C.
6: #C C drives the car with the steering wheel in both hands.
7: #C C drives the car with his right hand on the steering wheel.
8: #C A car drives past C.
9: #C C drives the car on the road with his left hand on the steering wheel
7b9303a2-c1be-43a1-9aa1-69325714b6d6.mp4
0: #C A car drives past C.
1: #C C drives past a bike on the road
2: #C A car drives past C.
3: #C A black car drives past C.
4: #C A car drives past C.
5: #C A car drives past C.
6: #C A motorcycle rides past C.
7: #C A blue car drives past C.
8: #C A truck drives past C.
9: #C A car drives past C.
c02fde47-3ef5-43ba-b673-b567ee30c6e4.mp4
0: #C C looks to the right
1: #O person A turns on the car lights
2: #O A man X drives the car
3: #C C looks around the road 
4: #O The man X holds the steering wheel with his left hand
5: #C C looks around the road
6: #C C turns left on the road.
7: #C C looks on the road
8: #C C holds a steering wheel
9: #C C looks to the left
bcee485f-1a72-44db-af87-5afdd0a592af.mp4
0: #C C moves the hand on the steering wheel
1: #O A man Z interacts with C.
2: #C C drives the car with the steering wheel in his right hand.
3: #C C drives the car on the road with his right hand.
4: #O The man A holds the phone with his right hand.
5: #C C stops the car.
6: #C C drives the car
7: #C C moves his left hand on the steering wheel.
8: #C C drives the car on the road
9: #C C drives the car
3e05f4f1-1d70-49f8-b1c5-a73351e04d52.mp4
0: #C C looks at the road 
1: #O The man Z operates the phone with his left hand.
2: #O The man A operates a phone in his right hand.
3: #C C drives a car
4: #O A man X operates a phone with both hands.
5: #C A white car drives past C.
6: #O The man A holds a phone with his right hand.
7: #C C drives the car
8: #O The man A operates the phone with his right hand.
9: #O person B operates phone
bd26fa34-cfd0-4b70-8f06-d6ef24ce2e90.mp4
0: #C C drives the car with the steering wheel in both hands.
1: #C C drives the car with his left hand.
2: #C A bus drives past C.
3: #C A black car drives past C.
4: #C A yellow car drives past C.
5: #C C drives the car on the road with his right hand on the steering wheel
6: #C C holds the steering wheel with his left hand.
7: #C C drives the car
8: #C C drives the car
9: #C A motorcycle rides past C.
c4d041b0-3efb-4a08-ab22-fea69657047a.mp4
0: #O The man X holds the steering wheel with both hands.
1: #O The man X drives the car on the road with his right hand
2: #C C looks around
3: #C C looks around
4: #C C looks around
5: #C A man K rides a bicycle past C.
6: #O A man B holds a phone with his left hand.
7: #C C looks around
8: #O The man X turns the steering wheel with both hands.
9: #C C looks around
0f25cd31-c827-46e0-9c05-138795ac60dc.mp4
0: #C A car drives past C.
1: #O A man B drives the car on the road with both hands.
2: #O A man E drives the car with the steering wheel in his left hand
3: #O A man Z drives a car on the road. 
4: #O A man D operates a phone with both hands.
5: #O The man A holds the phone with his left hand.
6: #O A man B drives the car with his right hand.
7: #C C drives past a car.
8: #C A white car drives past C.
9: #O A man P drives the car on the road with both hands.
733193dd-d5ba-42b9-8cf6-06c002a58088.mp4
0: #O A man X drives the car
1: #O A woman X drives the car
2: #C C holds the steering wheel with her left hand.
3: #O The man B removes his right hand from the steering wheel of the car
4: #C C looks around the car
5: #C C drives the car with both hands.
6: #C C turns a steering wheel with his left hand.
7: #C A yellow car drives past C.
8: #O The woman Y holds the car steering wheel with her right hand.
9: #C C looks at the side mirror
6f3add3f-a214-40d6-8c25-789f1700977b.mp4
0: #C C drives the car with his right hand.
1: #C C drives a car
2: #O A white car drives past C.
3: #C C drives the car on the road with his right hand.
4: #C C holds the steering wheel with his right hand.
5: #C C drives the car with his right hand.
6: #C C turns to the side
7: #C C turns car steering 
8: #O The man B operates the phone with his right hand.
9: #C C looks at the road 
1347222b-22b0-4994-95b5-128a4f1d10f4.mp4
0: #O A man Y operates a phone
1: #C C looks around 
2: #O person A holds a phone
3: #O A man Y operates a phone
4: #C C drives a car 
5: #O The man A operates the phone with both hands
6: #C C drives the car with his left hand.
7: #C C looks around the road
8: #O A man X uses a phone
9: #C C looks around
dd9de7d8-7f69-4e33-a64a-d95235993b24.mp4
0: #C C turns right
1: #C C looks around
2: #O The man X holds the phone with his right hand
3: #O The man X holds the phone with his left hand
4: #O person X uses phone
5: #C C looks around
6: #O The man A operates the phone with his left hand
7: #C C looks around 
8: #C C looks around
9: #O man X uses phone
7ddb7ff3-00a5-4b98-a591-dabc1499b914.mp4
0: #O A man X operates a phone
1: #O The man D operates the phone with his right hand
2: #O A man X operates a phone in his right hand
3: #C C drives the car
4: #O Person B operates phone
5: #C C looks around
6: #C C drives car on the road
7: #O Person A uses a phone
8: #C C drives a car 
9: #O The man X operates the phone with his left hand
4c227b43-0284-422a-8f7a-3124bfe3821d.mp4
0: #C C turns the car to the left.
1: #C C looks around 
2: #O The man A operates the phone with both hands.
3: #C C looks around
4: #O person X uses a phone. 
5: #C C looks around
6: #O The man X operates a phone with his right hand
7: #C C looks around
8: #O A woman Z operates a phone with both hands.
9: #C C interacts with a man Y.
3fb47f45-c977-46a7-9e7e-8c83e54d7ab4.mp4
0: #C C drives the car with both hands.
1: #C C drives a car 
2: #C C drives the car with his right hand on the steering wheel
3: #C C turns the steering wheel with his right hand.
4: #C C turns the steering wheel with both hands.
5: #C A motorcycle rides past C.
6: #C A red car drives past C.
7: #C C holds the steering wheel with both hands.
8: #C C drives the car
9: #C C turns the car steering with his right hand.
0e5e1c6b-4147-47c6-b97a-0f64f4c57d1c.mp4
0: #C C drives the car on the road with both hands.
1: #C C drives the car with his right hand.
2: #O The man B presses the phone in his right hand.
3: #C C looks around
4: #O The man D holds the phone with both hands.
5: #O A man X drives the car
6: #C C holds the steering wheel with his right hand.
7: #C C drives past a motorcycle.
8: #C C turns the steering wheel with both hands.
9: #O The woman X holds the steering wheel with her right hand
fe3df83d-8d36-404a-848e-e3dae1bc44ea.mp4
0: #C a motorcycle rides past C.
1: #C a white car drives past C.
2: #C a motorcycle drives past C.
3: #C a motorcycle rides past C.
4: #C C drives past a bike on the road.
5: #C C drives past a blue car.
6: #C C removes his left hand from the steering wheel
7: #C A motorcycle rides past C.
8: #C C drives past a car.
9: #C C drives past a car.
db73ddf0-6e1b-45db-9a1e-6cb73e2cdbc5.mp4
0: #O A man B operates a phone in a car with both hands
1: #O person B operates a phone
2: #O The man A operates the phone with his right hand.
3: #C C looks at a man X 
4: #O A man Y uses a phone 
5: #C C looks around in a car.
6: #O A man Y operates a phone 
7: #C C looks at a vehicle
8: #O A man Y operates a mobile phone with his right hand.
9: #O The man B operates a phone in his right hand.
2d20e294-3dbc-4e5c-a7bf-4b7b735695d8.mp4
0: #O The man B operates the phone with his right hand
1: #O A man X holds a phone with his left hand
2: #C C turns car steering wheel
3: #O The man A operates the phone with his left hand
4: #O A man Z drives a car on the road.
5: #C C looks around 
6: #C C drives the car with his left hand.
7: #C C drives the car
8: #C C looks around
9: #O The man A operates a phone with his left hand
3b167af5-93cc-42d0-8861-c3a4bd20159f.mp4
1: #C A car drives past C.
2: #C C drives past a car.
3: #C C turns the car steering with his right hand.
4: #C C drives the car with the steering wheel in both hands.
5: #C C drives the car on the road with his right hand.
6: #C C turns the steering wheel with his right hand.
7: #C A motorcycle rides past C.
8: #C A car drives past C.
9: #C A car drives past C.
4789e7d0-77fb-47a4-9526-68c9a06562f4.mp4
0: #C A motorcycle rides past C.
1: #O man Y uses the phone 
2: #C C looks around
3: #O The man A operates the phone in his left hand with his right hand
4: #C C drives the car with the steering wheel in her right hand.
5: #O The man A operates the phone with both hands.
6: #O The man X operates a phone with his left hand.
7: #C C looks around
8: #C C drives the car
9: #O A man Y operates the phone with his right hand.
a5ec060c-5901-492e-bac7-3e91c49d3717.mp4
0: #C C holds the steering wheel with his left hand.
1: #C C holds the steering wheel with his right hand.
2: #C C drives the car with his right hand. 
3: #C A white car drives past C.
4: #C A car drives past C.
5: #C A car drives past C.
6: #C C drives past the blue car.
7: #C A black car drives past C.
8: #C A car drives past C.
9: #C A white car drives past C.
ab9c2b30-e851-4019-9af6-2ffd244c6f58.mp4
0: #O The woman B operates the phone with both hands.
1: #C A motorcycle rides past C.
2: #O The woman B operates the phone with her right hand
3: #O The woman B operates the phone with her left hand
4: #C A motorcycle rides past C.
5: #O The woman D operates a phone with both hands.
6: #C A motorcycle rides past C.
7: #O The woman A operates the phone in her left hand
8: #C C drives past a man B.
9: #C A black car drives past C.
5206951b-4d71-46ba-81a3-312e77c79257.mp4
0: #O person X drives the car
1: #C C looks around
2: #C C looks at the side mirror
3: #C C looks around
4: #O The man X holds the phone with his right hand
5: #O person B operates phone
6: #O The man A operates the phone with both hands.
7: #O A man Y holds a phone
8: #C A black car drives past C.
9: #O Person A holds the phone
0f67cd91-8587-498b-bd18-8267991d7930.mp4
0: #C C turns car steering
1: #C C drives the car on the road with his right hand.
2: #C C turns a steering wheel with his right hand.
3: #C C drives the car on the road with his right hand.
4: #C C drives past a blue car.
5: #C C drives the car on the road with his right hand on the steering wheel
6: #C C drives a car on the road
7: #C C drives the car with the steering wheel in both hands.
8: #C C drives the car on the road with his right hand.
9: #C C turns car right
487663b3-505c-41ed-894c-e49693388349.mp4
0: #O The woman B operates a phone with both hands.
1: #C C holds a phone in her right hand.
2: #C A motorcycle rides past C.
3: #C C drives past a black car.
4: #O The woman X operates the phone with her right hand.
5: #O The woman B operates a phone with both hands.
6: #O The woman A holds a phone with her right hand.
7: #O The woman B holds a phone with both hands.
8: #O The woman A operates the phone with her right hand.
9: #O The woman B operates the phone in her left hand with her right hand.
7eac0ed0-88e0-48e9-843e-ac7d81eccd0b.mp4
0: #O The woman B holds the phone with her right hand.
1: #O The woman A operates the phone with her right hand.
2: #C C holds the steering wheel with his right hand.
3: #C C holds the steering wheel with his right hand.
4: #O The woman A presses the phone with her right hand.
5: #C C holds the steering wheel with his left hand.
6: #C C drives past a car.
7: #O The woman A holds the phone with her left hand.
8: #C C drives the car with the steering wheel in his right hand
9: #O The woman A drives the car on the road with both hands
cffd6e74-9d43-45c1-ac75-8621ce367350.mp4
0: #C C moves his right hand on the gear stick.
1: #C C drives past a bike on the road.
2: #C C drives the car with the steering wheel in both hands.
3: #C A motorcycle rides past C.
4: #C C drives the car with his right hand.
5: #C C drives past a motorcycle.
6: #C C drives past a man L on the motorcycle.
7: #C C drives past a bike on the road.
8: #C C turns the steering wheel with his left hand.
9: #C C drives past a motorcycle.
246dce98-5189-45d4-bbe9-aebc664b1568.mp4
0: #C C turns the steering wheel with his right hand.
1: #C C drives the car with his right hand.
2: #C C turns the steering wheel with his left hand.
3: #C C turns the steering wheel with his right hand.
4: #C C drives the car with the steering wheel in both hands.
5: #C C turns the steering wheel with his right hand.
6: #C C drives the car with his right hand.
7: #C C drives the car with the steering wheel in his right hand.
8: #C C drives the car with his right hand on the steering wheel.
9: #C C drives the car with the steering wheel in his right hand.
9995a4a3-bafe-42ac-8d45-c0bafe8cfe54.mp4
0: #C C drives the car on the road
1: #C C Drives a car on the road
2: #C C drives the car 
3: #C C drives the car with his right hand on the steering wheel
4: #C C drives the car with his right hand on the steering wheel
5: #C C turns right on the road
6: #C C turns right
7: #C C drives the car on the road.
8: #C C turns the car steering wheel with his right hand.
9: #C C turns right on the road
37d8d912-94a7-4649-b76b-5fe0305449da.mp4
0: #O The woman X drives past a white car.
1: #C A white car drives past C.
2: #C A black car drives past C.
3: #C A car drives past C.
4: #O A car drives past C.
5: #O The woman Y drives the car with both hands on the steering wheel.
6: #C A car drives past C.
7: #C C drives past a blue car.
8: #C C drives past a white car.
9: #C A red car drives past C.
3b11fa4e-4c6a-4925-aedc-ab7c536bc565.mp4
0: #C C drives the car with both hands on the sterring wheel
1: #C C holds the steering wheel with his left hand.
2: #C C drives the car with both hands.
3: #C C drives the car with both hands on the steering wheel.
4: #C C drives past a yellow car.
5: #C C holds the steering wheel with his right hand.
6: #C C drives past a bike on the road.
7: #C C drives the car with his right hand.
8: #C C turns the steering wheel with both hands.
9: #C C drives the car on the road with both hands.
13e3a338-50ad-4c2c-acdb-98daa8277ac0.mp4
0: #C C drives the car with his right hand on the steering wheel.
1: #O A man D drives a car on the road with both hands.
2: #C A car drives past C.
3: #O A man Y drives a car
4: #O The man B holds the steering wheel with his right hand.
5: #C C moves his right hand on the steering wheel.
6: #C A car drives past C.
7: #O A car drives past C.
8: #C C drives the car on the road with his right hand on the steering wheel.
9: #C A blue car drives past C.
fd189f02-ca1b-4fbc-8d89-b0be8bc371c5.mp4
0: #C C moves the gear lever with his left hand.
1: #C C turns the gear lever with his left hand.
2: #C C controls steering wheel   
3: #C C drives the car with both hands on the sterring wheel.
4: #C C drives the car
5: #C C drives the car with the steering wheel in his left hand.
6: #C C drives the car on the road with his left hand.
7: #C C presses a button in the car with his left hand.
8: #C C adjusts a knob on the car with his left hand.
9: #C C drives past a truck.
a824f6c0-5410-4c0c-bc67-28a39c55a04f.mp4
0: #C C drives past a motorcycle.
1: #C A motorcycle rides past C.
2: #C C looks around
3: #C A motorcycle rides past C.
4: #C C drives the car
5: #C C drives the car
6: #C A motorcycle rides past C.
7: #C C drives the car with his right hand.
8: #C C looks around
9: #C C drives the car with his right hand.
15a45b00-a011-4a0c-8bec-a9c71df136d7.mp4
0: #O The man A holds the steering wheel with his left hand.
1: #O The man B holds the steering wheel with his left hand.
2: #O The man B removes his right hand from the steering wheel.
3: #O The man X moves the car gear shift with his right hand. 
4: #C A black car drives past C.
5: #O A black car drives past C.
6: #C C drives the car with his right hand.
7: #C A white car drives past C.
8: #O A man X holds a phone with both hands.
9: #C A black car drives past C.
d39b2e99-0194-49ab-abb2-7e984cec2085.mp4
0: #C C drives the car with his right hand.
1: #C C drives the car on the road with the steering wheel in both hands.
2: #C A red car drives past C.
3: #C A car drives past C.
4: #C C drives the car on the road with both hands.
5: #C C drives the car on the road with the steering wheel in his left hand.
6: #C C drives the car on the road with his right hand on the steering wheel.
7: #C A car drives past C.
8: #C C drives the car with his right hand.
9: #C A car drives past C.
f37186fd-6914-467b-bc02-418653fa8f55.mp4
0: #C C looks through the car mirror
1: #O A man Z touches his head with his right hand.
2: #O The man X touches his head with his right hand.
3: #O The man X adjusts a helmet on his head with his left hand.
4: #C C drives past a red car.
5: #C C moves a hand
6: #C C looks at man X 
7: #O The man Y adjusts his glasses with his right hand.
8: #O The man X adjusts a window shade with his left hand.
9: #O The man X drives the car with his right hand on the steering wheel.
6a92bcd6-8fec-4204-af0c-a5ee3c44e0d3.mp4
0: #O The man D holds the phone with his right hand.
1: #O The woman D holds the phone with her left hand.
2: #O The man A holds the phone with his right hand.
3: #O The woman Y holds the phone with her left hand.
4: #O A man X operates a phone with his right hand.
5: #O The man B operates the phone with his left hand.
6: #C A man Y rides a bicycle on the road past C with both hands
7: #C C drives the car with his right hand.
8: #C C drives the car with both hands.
9: #O The man X operates the phone with his left hand.
543afc20-6544-4248-b0d2-500c4e8fbe45.mp4
0: #C C looks around the road
1: #C C looks at the windscreen
2: #O person A turns to the right
3: #C C Looks around a road
4: #C C looks at the road
5: #O Person A stops the car
6: #O A man Z uses a phone
7: #O person X operates a phone.
8: #C C looks around the road
9: #O Person A holds a phone
72d10288-2ee3-4990-ad02-3c1e49fe5716.mp4
0: #C C holds the steering wheel with his right hand.
1: #C C drives past a motorcycle.
2: #C C turns on the car's sun visor with his left hand.
3: #C C drives the car on the road with his left hand on the steering wheel
4: #C C looks around
5: #C C looks around
6: #C C drives the car with his right hand.
7: #C C places his right hand on the car steering wheel.
8: #C C drives the car
9: #C A white car drives past C.
95a32845-3d7c-4700-a6d5-b55bedd802dd.mp4
0: #C C drives past a truck.
1: #C C drives the car with his right hand on the steering wheel.
2: #O man Y uses phone 
3: #O The woman A holds the phone with both hands.
4: #C A white truck drives past C.
5: #C C drives past a red bus.
6: #C C drives past a blue bus.
7: #C C operates the gear shift with his left hand.
8: #C C operates the phone with his right hand.
9: #C C drives the car on the road with his right hand on the steering wheel.
ddde062c-c2b1-43a5-b57a-6c6dea429e10.mp4
0: #C C holds a steering wheel with his left hand.
1: #C A white truck drives past C.
2: #C C holds a steering wheel with his right hand.
3: #C C turns the steering wheel with his right hand.
4: #C C holds the steering wheel with his right hand.
5: #O A man A holds a phone with his right hand.
6: #C C drives the car
7: #O The woman X operates the phone with her right hand.
8: #C C holds the steering wheel with his right hand.
9: #C C drives past a truck.
3a666b54-c08a-4388-afb1-01321c26399f.mp4
0: #C C holds the gear shift with his right hand.
1: #C A car drives past C.
2: #O The man X presses the button of the car dashboard with his right hand
3: #O The man A drives the car with his left hand.
4: #O The man B holds the car steering with his right hand.
5: #O The man X holds the car steering with his right hand.
6: #O The man X opens a car door with his left hand.
7: #C C stops the car
8: #O A man X drives the car with both hands.
9: #O The man X touches the car gear with his left hand.
7ecce060-636d-480f-8e1b-aad7f4725725.mp4
0: #C C drives the car on the road with both hands on the steering wheel
1: #C C drives the car with his right hand.
2: #C C turns the steering wheel with both hands.
3: #C C drives the car with his right hand on the steering wheel.
4: #C C turns the steering wheel with both hands.
5: #C C drives the car with both hands on the steering wheel.
6: #C A white car drives past C.
7: #C C drives past a car.
8: #C C drives the car with the steering wheel in his right hand.
9: #C C drives past a yellow car.
d309efb4-6ecb-40e0-9914-b2477139134f.mp4
0: #C C drives the car on the road with his left hand on the steering wheel
1: #O person A operates phone
2: #C A white car drives past C.
3: #O A man Z operates a phone in his right hand.
4: #O The man A holds the phone with his right hand.
5: #O A black car drives past the man A.
6: #O A man B operates a phone with his left hand.
7: #O A man X operates a phone in his left hand.
8: #C A car drives past C.
9: #O The man A operates a phone with both hands.
7ffd3799-143f-4053-ad4d-9bc0cb61a9e3.mp4
0: #O The man X drives the car on the road with his right hand on the steering wheel
1: #O A man Y drives a car
2: #C C places his right hand on his lap.
3: #C A black car drives past C.
4: #C C drives the car with his right hand.
5: #C C holds a car steering with his right hand.
6: #O The man A holds the steering wheel with his right hand.
7: #C C drives the car with his left hand.
8: #C A man B rides a bicycle on the road past C.
9: #O The man X drives the car with the steering wheel in his hands.
2fa7c425-ac52-411c-af2e-bb781317bc0d.mp4
0: #C C drives past a bike on a bike trailer.
1: #C A car drives past C.
2: #C A motorcycle rides past C.
3: #C A black car drives past C.
4: #C A motorcycle rides past C.
5: #C A black car drives past C.
6: #C A white car drives past C.
7: #C A motorcycle rides past C.
8: #O The woman B presses the phone in her right hand
9: #C A motorcycle rides past C.
b2490404-8ee5-4970-9839-b0bdab89703e.mp4
0: #C C drives car on the road
1: #C C drives past a motorcycle.
2: #O a person M uses phone
3: #C A motorcycle rides past C.
4: #C C drives the car with both hands.
5: #C C drives the car with both hands.
6: #C C drives the car with his right hand.
7: #C A black car drives past C.
8: #C A motorcycle rides past C.
9: #C C drives the car with both hands on the steering wheel.
c8af615b-f689-43a4-9bed-4aeea7ae07a5.mp4
0: #O The man X operates the phone with both hands.
1: #O The man Y operates the phone with both hands.
2: #C C looks around
3: #O The man X holds the steering wheel with his right hand.
4: #O The man A turns the steering wheel with his right hand.
5: #C C turns right
6: #C C drives the car with both hands on the steering wheel.
7: #O The man X operates the phone with his right hand.
8: #O The man A operates the phone with both hands.
9: #C C looks around
d1f3baba-0b15-4351-8997-d902f4669481.mp4
0: #C C looks around the car
1: #C C turns towards a man Y.
2: #O A man P drives the car.
3: #O The man X adjusts the camera on his head with both hands.
4: #C A white car drives past C.
5: #O A man X drives the car
6: #O A black car drives past the car.
7: #O A white car drives past C.
8: #C C looks at the road
9: #C C touches a car with his right hand.
30c65afb-93f4-4857-bcaf-cb2b0e57c9a7.mp4
0: #O A man X adjusts a face mask with his left hand.
1: #C A white car drives past C.
2: #C C looks at man X
3: #C C drives the car with his right hand.
4: #O man X drives the car
5: #O The man A holds the steering wheel with his left hand.
6: #O The man X touches the car gear with his right hand.
7: #C C looks around
8: #C C drives the car with his right hand on the steering wheel
9: #C C drives the car
649efb4d-8923-4ad1-9741-018b8e504bcb.mp4
0: #C A motorcycle drives past C.
1: #C C drives past a car.
2: #O A motorcycle rides past C.
3: #C C drives past a motorcycle.
4: #C A white car drives past C.
5: #C C drives past a yellow car.
6: #C A motorcycle rides past C.
7: #C C drives past a yellow car.
8: #C C drives past a motorcycle.
9: #O A man G operates a phone with his right hand
81c717f5-d373-46c7-a011-bb751192941a.mp4
0: #O A man J operates a phone with his left hand.
1: #C C drives past a car.
2: #C C holds the steering wheel with his right hand.
3: #C C drives the car with his right hand.
4: #C C drives the car with his right hand.
5: #C C holds a steering wheel with his left hand.
6: #C A man J rides a bicycle past C.
7: #C C drives the car on the road. 
8: #C C turns the steering wheel with his right hand.
9: #C C Drives a car on a road
5f149e91-dc4e-46a1-b2c6-acb92e0fc109.mp4
0: #C C drives past a car.
1: #C C drives past a white car.
2: #C A white car drives past C.
3: #C C drives past a white car.
4: #C C holds the steering wheel with his left hand
5: #C C drives past a red car.
6: #C C holds the steering wheel with his right hand
7: #C C removes his right hand from the steering wheel
8: #C C turns the steering wheel with his right hand
9: #C C holds the steering wheel with his left hand
f77132f9-cae2-4d60-8f92-7f6cafc00589.mp4
0: #O The man A operates the phone with both hands.
1: #O The man A holds the phone with his left hand.
2: #O A man X operates a phone in his right hand.
3: #O The man A presses the phone with his right hand.
4: #O The man A operates a phone with his right hand.
5: #O A man X holds a phone with his right hand.
6: #C C drives the car on the road with his right hand on the steering wheel
7: #O The man B operates the phone with his right hand.
8: #C C looks around
9: #O A man Y operates a phone with his right hand.
94aecbcf-5b0a-4449-b060-483f8f189a72.mp4
0: #C C looks around 
1: #C C looks around
2: #C C looks at the road
3: #C C looks around 
4: #C C looks to the right
5: #C C turns to the right
6: #O The man X drives the car with the steering in his right hand
7: #O A man X drives a car on the road
8: #C C drives the car with his right hand.
9: #O A man Y drives the car 
5af674e2-a4b0-4e8f-894f-55e0cb004978.mp4
0: #C A car drives past C.
1: #C C drives past a yellow car.
2: #C A white car drives past C.
3: #C C drives the car with the steering wheel in both hands.
4: #C C drives the car with the steering wheel in his right hand
5: #C A car drives past C.
6: #C A car drives past C.
7: #C A white car drives past C.
8: #C A car drives past C.
9: #C A black car drives past C.
00a5df39-9d21-451d-823b-546d5a9eee13.mp4
0: #C C drives the car with his left hand on a steering wheel
1: #C C drives past the white car.
2: #C C drives past the truck.
3: #C C drives the car with the steering wheel in his right hand
4: #C C drives past a car. 
5: #C C holds the steering wheel with his right hand.
6: #C C drives past a man F.
7: #C C drives past a yellow car.
8: #C C drives past a motorcycle.
9: #C A motorcycle rides past C.
e0c6c7a3-420a-4d30-a633-c0b1bdc85a60.mp4
0: #C C drives the car on the road with both hands.
1: #C C drives the car
2: #C C drives the car with his right hand.
3: #C C turns right on the road.
4: #C C drives the car with both hands.
5: #C C drives the car
6: #C C turns the car to the left
7: #C C turns right on the road.
8: #C C drives the car on the road with his right hand on the steering wheel
9: #C C looks around
13cb19b8-7d2b-4771-8fa9-2e16de9f2734.mp4
0: #C A man S rides a bicycle past C
1: #C C drives past a motorcycle.
2: #C A white car drives past C.
3: #C A motorcycle rides past C.
4: #C A motorcycle rides past C.
5: #C C drives past a man L.
6: #C A motorcycle rides past C.
7: #C A motorcycle rides past C.
8: #C A motorcycle rides past C.
9: #C A motorcycle rides past C.
42467492-9ca2-4e52-b63c-f04bd6518d61.mp4
0: #O Person A operates a phone
1: #O The man X operates the phone with his right hand
2: #C C looks around
3: #O A man Y operates a phone 
4: #O person Y looks around
5: #C C looks at the car
6: #O person B operates the phone
7: #O person A operates a phone
8: #O A man Y holds a phone
9: #O person B operates phone
8c696e60-dcd2-44c1-add9-88d11b5ab4bb.mp4
0: #C C drives the car 
1: #C C drives the car 
2: #C C drives car on the road
3: #C C drives the car 
4: #C C looks around
5: #C C drives the car with the steering wheel in his left hand.
6: #C C turns right on the road.
7: #C C looks around
8: #C C drives the car with the steering wheel in both hands.
9: #C C looks around
9f0ae307-0e45-421d-936c-f3ea687b553f.mp4
0: #C A motorcycle rides past C.
1: #C A motorcycle rides past C.
2: #C A black car drives past C.
3: #C C drives past a motorcycle.
4: #C A motorcycle rides past C.
5: #C A white car drives past C.
6: #O A man D rides a bicycle on the road with both hands.
7: #O A man B rides a bicycle on the road with his right hand
8: #C A motorcycle rides past C.
9: #C A black motorcycle rides past C.
f154870c-cd1d-4b68-afe8-00de2016b899.mp4
0: #O The woman A drives the car with the steering wheel in both hands
1: #C C drives the car with the steering wheel in his left hand.
2: #C C drives the car with his left hand on the steering wheel.
3: #C C drives the car with the steering wheel in both hands.
4: #C C drives the car with the steering wheel in his right hand.
5: #O The woman B operates the phone with both hands.
6: #O The man B holds the phone with his right hand.
7: #C A car drives past C.
8: #O The woman B holds the phone with both hands.
9: #O The man X operates the phone with his right hand.
e6d9e6c2-adc5-4ba7-9527-dd7f6bfc2043.mp4
0: #O person X drives the car
1: #C C drives the car
2: #C C holds the steering wheel with his right hand.
3: #O Man X drives car
4: #C C drives the car with his right hand on the steering wheel.
5: #O person X drives the car
6: #O person A points at a phone
7: #C C drives the car on the road
8: #O The man X operates a phone with his right hand.
9: #O The man A holds the phone with his right hand.
aadf4865-0ee3-40f6-90f6-3d7e305d317a.mp4
0: #C C turns right 
1: #C C turns the car to the left 
2: #C C drives the car
3: #C C Drives a car on a road
4: #C C moves a hand
5: #C C drives a car
6: #C C drives the car
7: #C C drives a car
8: #C C turns right 
9: #C C looks around 
a3655e07-6ed9-4307-a869-5adc3575e141.mp4
0: #O The man A operates a phone with both hands.
1: #C C looks around
2: #C C drives the car with his right hand on the steering wheel
3: #C C drives the car with his right hand.
4: #C C drives a car
5: #C C drives a car on the road
6: #C C drives the car 
7: #C C drives car on the road
8: #C C drives the car
9: #C C looks around
c2254f6d-d6d9-4765-b5db-9a9be1b10bd7.mp4
0: #C C stares at a phone
1: #C C looks around 
2: #C C looks around the road
3: #C C drives the car
4: #C C Looks around a parking lot
5: #C C turns a steering wheel with both
6: #C C looks around
7: #C C turns right
8: #C C Looks around a parking lot
9: #C C looks at the road
5e49a8e1-020d-4afa-8fec-49dc8bfd381f.mp4
0: #O A man X operates a phone with both hands.
1: #C A car drives past C.
2: #O The man X operates the phone with both hands.
3: #C C looks around
4: #O The man A places his left hand on his face. 
5: #O The man A operates the phone in his left hand.
6: #O The man A operates the phone with his right hand.
7: #O The man X drives the car with the steering in both hands.
8: #O A man Y operates a phone with his right hand.
9: #C A white car drives past C.
cc6ed76b-d98f-4a23-812e-29fe6c828948.mp4
0: #C A white car drives past C.
1: #C C drives past a car.
2: #C C drives past a car.
3: #C A car drives past C.
4: #C C drives past a motorcycle.
5: #C C drives past a yellow car.
6: #C A yellow car drives past C.
7: #C A motorcycle rides past C.
8: #C C drives past a car.
9: #C A motorcycle rides past C.
becf970e-4a71-4cdb-89c0-75d824eb472e.mp4
0: #C A motorcycle rides past C.
1: #C C drives past a motorcycle.
2: #O A man X operates a phone with his right hand
3: #C C drives past a motorcycle.
4: #O The man X operates a phone with his right hand
5: #C C drives past a motorcycle.
6: #O A man B operates a phone with his right hand
7: #C C drives past a motorcycle.
8: #C C drives past a truck.
9: #C A car drives past C.
6a258be7-7b89-4e36-bfbf-553856b77601.mp4
0: #C A motorcycle drives past C.
1: #C C drives past a motorcycle.
2: #O The man A operates the phone with his right hand
3: #O A motorcycle drives past C.
4: #C A motorcycle drives past C.
5: #O The man X operates the phone with both hands.
6: #C C drives past a white car.
7: #O A car drives past C.
8: #C C drives past a man L on a bicycle.
9: #C C drives past a motorcycle.
f98dc337-ad0d-4328-bd36-b317ba2e0dd8.mp4
0: #O The man A holds the phone with his right hand.
1: #O The man Y operates the car gear with his right hand.
2: #O man X uses his phone
3: #C C drives car on the road
4: #O A man X uses phone 
5: #C C looks around
6: #C A motorcycle rides past C.
7: #O A man X operates a phone 
8: #O A man X holds the phone with both hands
9: #C C turns left on the road
b0f54954-2761-4b24-a902-8e84cd1e8e1f.mp4
0: #C C drives the car with his left hand on the steering wheel
1: #C C holds the steering wheel with his left hand.
2: #C C drives the car with his right hand.
3: #C C drives past a yellow car.
4: #C C turns the steering wheel with his left hand.
5: #C C drives past a motorcycle.
6: #C C moves the gear shift with his left hand.
7: #C A black car drives past C.
8: #C C drives past a woman Y.
9: #C C drives past a white car.
3c3d70fc-5621-48fc-80f1-682d2fc561ef.mp4
0: #C A car drives past C.
1: #O A man S drives the car on the road with his right hand
2: #C A car drives past C.
3: #O A man Y operates a phone 
4: #C A car drives past C.
5: #C A black car drives past C.
6: #C A car drives past C.
7: #C C looks around
8: #C A car drives past C.
9: #O A man X operates a phone with both hands.
b80414d8-2dd5-4c43-afd7-7664fdf19468.mp4
0: #C C drives past a white car.
1: #C A motorcycle rides past C.
2: #C C holds the steering wheel with both hands.
3: #C C holds the steering wheel with his left hand.
4: #C C drives past a truck.
5: #C C drives past a car.
6: #C C drives past a car.
7: #C C drives past a motorcycle.
8: #C C drives past a motorcycle.
9: #C C drives past a motorcycle.
ba6ac3a6-f081-491a-a95c-7e95ff731ade.mp4
0: #C A motorcycle rides past C.
1: #C C drives past a motorcycle.
2: #C C drives the car with both hands.
3: #C A motorcycle rides past C.
4: #C C drives past a motorcycle.
5: #C C drives past a bicycle.
6: #C A motorcycle rides past C.
7: #C C drives past a car.
8: #C C drives the car with his right hand.
9: #C C turns the steering wheel with his right hand.
cee71821-5f8f-483e-b019-9e2840677caf.mp4
0: #C C drives car on the road
1: #C C drives a car on the road
2: #C C holds the car's steering wheel with his left hand.
3: #C C drives the car 
4: #C C drives a car 
5: #C C drives the car
6: #C C drives the car
7: #C C drives a car
8: #C C drives car on the road
9: #C C drives the car with both hands.
cc7784e7-79e8-4d7e-8fd6-1bc475e9ecb0.mp4
0: #C C drives past a car.
1: #C C drives past a car.
2: #C C drives past a car. 
3: #C C drives past a truck.
4: #C C drives the car
5: #C C drives past a car.
6: #C C drives the car with the steering wheel in his right hand
7: #C A black car drives past C.
8: #C C drives past a truck.
9: #C C drives the car with both hands.
1e84778e-9856-44de-b1eb-6ffffd6318a1.mp4
0: #C C drives the car with both hands.
1: #C C drives the car with the steering wheel in both hands.
2: #C C holds the steering wheel with his right hand.
3: #C C turns the steering wheel with his right hand.
4: #C C drives the car with the steering wheel in his left hand.
5: #C C drives the car with the steering wheel in his left hand.
6: #C C drives the car with his right hand.
7: #C C removes his right hand from the steering wheel.
8: #C C holds the steering wheel with his right hand.
9: #C C drives the car on the road with his right hand.
e335bdad-6693-4583-8479-38c894e4c94a.mp4
0: #C A motorcycle rides past C.
1: #C A motorcycle rides past C.
2: #C A motorcycle rides past C.
3: #C A motorcycle rides past C.
4: #C C drives past a bike on the road.
5: #C C drives past a car.
6: #C C drives past a motorcycle.
7: #C A motorcycle rides past C.
8: #C C drives past a motorcycle.
9: #C A motorcycle rides past C.
bec4a754-dab9-4071-88fc-110d4f5afd72.mp4
0: #O Person A holds steering wheel with right hand
1: #O A man X drives the car
2: #C C turns left
3: #C C turns the car steering.
4: #C C looks at the side mirror
5: #O The man X turns on the car's wiper with his right hand
6: #C C turns the steering wheel with his right hand.
7: #O man X holds steering wheel
8: #O The man X turns off the car's car light with his right hand
9: #O man X drives car on the road
c6e794ed-c824-44f4-ac9e-7a0f5c5d4ad9.mp4
0: #C C drives past a bus.
1: #C C drives past a white car.
2: #C C drives past a motorcycle.
3: #C C turns a steering wheel with his hands.
4: #C A yellow car drives past C.
5: #C A white car drives past C.
6: #C C drives past a car.
7: #C C drives past a red car.
8: #C A white car drives past C.
9: #C A white car drives past C.
b9ad4231-ebbf-47e3-aab0-106331c9d0e9.mp4
0: #C C drives a car 
1: #C C holds the steering wheel with his left hand.
2: #C C drives the car with both hands.
3: #C C drives the car
4: #C C turns the steering wheel with his right hand.
5: #C C turns the steering wheel to the right with both hands.
6: #C C holds the steering with left hand
7: #C C turns the steering wheel to the right with both hands.
8: #C C turns right 
9: #C C turns left on the road
95ec9048-cd47-4186-9fa2-27e040836de1.mp4
0: #C C Drives a car on the
1: #C C looks to the right
2: #C C looks on the other side
3: #C C turns left
4: #C C turns right
5: #C C looks on the right side
6: #C C drives the car 
7: #C C drives the car
8: #C C turns right
9: #C C drives the car on the road
af5ce11e-6ed9-42a1-b06b-912fb5bd5467.mp4
0: #O The man B presses the phone with his right hand
1: #O The woman X holds the phone with both hands.
2: #O The man B holds the phone with both hands.
3: #O The woman A operates the phone with both hands.
4: #O The man B operates the phone with his right hand
5: #O The woman A operates the phone with her right hand
6: #O The woman X operates the phone with both hands.
7: #O The woman A operates the phone with both hands.
8: #O The woman Y operates a phone with both hands.
9: #O The woman A holds the phone with both hands.
dad432f3-3802-4972-9386-4394a6da5482.mp4
0: #C C looks around
1: #O The man A operates the phone with his left hand.
2: #C C drives the car with both hands on the steering wheel.
3: #O The man X operates the phone with his left hand.
4: #C C drives past a motorcycle.
5: #C A black car drives past C.
6: #O The man B operates the phone with his right hand.
7: #O The man B operates the phone in his right hand with his right hand.
8: #C C drives the car on the road with the steering wheel in his right hand
9: #O A man X operates a phone with both hands.
95424284-542e-40cf-b318-92bddc88f9a5.mp4
0: #C C turns car steering
1: #C C drives the car with the steering wheel in his left hand.
2: #C C drives the car with his right hand.
3: #C C drives past the motorcycle.
4: #C C drives past a motorcycle.
5: #C C looks around
6: #C C drives the car
7: #C C drives the car with both hands.
8: #C C drives past a motorcycle.
9: #C A motorcycle rides past C.
a273b5a6-17a5-42d5-9929-afc09745b9a0.mp4
0: #C C holds the steering wheel with his right hand.
1: #C C drives the car with both hands.
2: #C C holds a steering wheel with his right hand.
3: #C C holds the steering wheel with his left hand.
4: #C C turns the steering wheel with his right hand.
5: #C C holds the steering wheel with his right hand.
6: #C C turns right.
7: #C C holds steering wheel
8: #C C drives the car on the road.
9: #C C holds the steering wheel with his left hand.
ee052344-991f-4c69-91ff-82faebd6df33.mp4
0: #C C drives past a car.
1: #C C drives the car on the road with his right hand
2: #C A car drives past C.
3: #C C drives past a man B.
4: #C A car drives past C.
5: #C A car drives past C.
6: #C A car drives past C.
7: #C C drives past a car.
8: #C A white car drives past C.
9: #C C drives the car on the road with his right hand
bffbddb6-f719-426b-ba10-ef07c31820ea.mp4
1: #C C drives the car with the steering wheel in both hands. 
2: #C A car drives past C.
3: #C A car drives past C.
4: #C C drives the car with his left hand.
5: #C C drives the car with his right hand.
6: #C C drives a car on the road
7: #C C drives the car
8: #C C drives the car with the steering wheel in his hands.
9: #C C drives the car
b77b4bb7-2331-49ae-8e61-1c07305d2bba.mp4
0: #C C drives the car
1: #C C turns the steering wheel to the left with her left hand
2: #C C drives car on the road
3: #C C drives the car with the steering wheel in his hands.
4: #C C looks around
5: #C C drives the car
6: #C C turns a car steering wheel with her right hand.
7: #C C turns a steering wheel with both hands.
8: #C C drives the car on the road with both hands.
9: #C C drives the car
922731b5-4590-4bb9-b3d7-044fc3643e4b.mp4
0: #C C drives past a car.
1: #C A motorcycle rides past C.
2: #C A white car drives past C
3: #C A car drives past C.
4: #C C drives past a car.
5: #C A tricycle rides past C
6: #C A truck drives past C.
7: #C A white car drives past C
8: #C a car drives past C.
9: #C A car drives past C.
40c2ebea-e358-48cf-9782-61150af72d2b.mp4
0: #C C drives the car on the road with the steering wheel in his right hand
1: #C C drives the car with the steering wheel in his right hand.
2: #C C drives the car with his right hand.
3: #C C drives the car with the steering wheel in his left hand.
4: #C C drives the car with his right hand.
5: #C C drives the car on the road with his right hand.
6: #C C drives the car with both hands on the steering wheel. 
7: #C C drives car on the road
8: #C C drives the car on the road.
9: #C C turns the steering wheel of the car with his left hand.
cda59804-6f66-4845-b5cc-511c4ee8d38e.mp4
0: #C C turns right
1: #O A man Y holds a phone
2: #O A man X drives a car
3: #O A man Y holds a phone
4: #C C Drives a car on the road
5: #O A man Z rides the bicycle
6: #C C drives the car on the road.
7: #C C drives on road
8: #O A man Y drives a car 
9: #O Person A turns right 
eb614f23-6f89-4b83-9403-aba5137a3f20.mp4
0: #C C looks around
1: #O The woman X adjusts her hair with her left hand
2: #C C looks around the car
3: #C C looks at man X
4: #C C looks at man X
5: #C C looks around
6: #O person Y uses the phone
7: #C C looks around
8: #C C looks at man X
9: #O person X looks at C
dd6d8cdc-a082-4d65-aac7-466c7e5bac8f.mp4
0: #C C drives past a truck.
1: #C C drives the car with his right hand on the steering wheel.
2: #C A motorcycle rides past C.
3: #O The man A holds the phone with his right hand.
4: #C C places his left hand on his lap.
5: #O The man A holds the phone with his right hand.
6: #C A motorcycle rides past C.
7: #O A man Y operates the phone with his right hand.
8: #O A man P operates a phone with his right hand.
9: #O The man B operates the phone with his right hand.
c81fd46d-5367-462f-aea7-29f38092195b.mp4
0: #O The man X holds the phone with his right hand.
1: #C C drives the car on the road with the steering wheel in his right hand.
2: #C C turns left
3: #C C looks around
4: #O A man Y drives the car
5: #C C looks around
6: #O A man X holds a phone with his right hand.
7: #O A Man X drives car on the road 
8: #C C turns to the side
9: #C C looks at the road
14114000-d5d2-41de-9a16-560ef251e5e8.mp4
0: #C C drives past the man K.
1: #C A motorcycle rides past C.
2: #C A car drives past C.
3: #O The man A drives the car with both hands on the steering wheel
4: #C C drives past a man S.
5: #C A motorcycle rides past C.
6: #O A yellow car drives past C.
7: #C A yellow car drives past C.
8: #O A car drives past C.
9: #C A man L rides a bicycle past C.
b4f9beda-dd4d-40d9-ad8e-ac08c640d508.mp4
0: #O A man Z uses phone
1: #C C drives past a yellow car.
2: #O The man A operates the phone with his right hand.
3: #O The man A operates the phone with both hands.
4: #O A man X drives the car
5: #C C turns to the side
6: #C C drives the car
7: #C A motorcycle rides past C.
8: #O The man X operates the phone in his right hand.
9: #O A man X sits on the car seat.
eadcb410-94e0-4548-9215-2e9efdd07da7.mp4
0: #C C Looks around a parking lot
1: #C C looks around
2: #C C looks around the car
3: #C C looks at the road
4: #C C looks around the car
5: #O a person M uses the phone
6: #O The man B operates the phone in his right hand
7: #O The man A operates a phone in his left hand
8: #O The woman B operates the phone with her right hand
9: #C C looks around
f4c2c7d9-893e-466b-96cc-28c92abb220c.mp4
0: #C A car drives past C.
1: #C A white car drives past C.
2: #O The man A places his right hand on his right lap.
3: #O The man B holds the steering wheel with his right hand. 
4: #C C turns the steering wheel with his right hand.
5: #C A yellow car drives past C.
6: #O The man X drives the car with his right hand on the steering wheel
7: #C A blue car drives past C.
8: #C A white car drives past C.
9: #C C drives past a bus.
c0b732cf-43a5-427f-a6c5-fcd6399d2efe.mp4
0: #C C looks at the road 
1: #C A yellow car drives past C.
2: #O The man X operates a car remote control with his right hand.
3: #O A black car drives past C.
4: #O The man B operates a phone in his right hand.
5: #C C holds the steering wheel with his left hand.
6: #C C moves the gear stick with his right hand.
7: #C C drives past a blue car.
8: #C C drives the car with the steering in his right hand.
9: #C A white truck drives past C.
bd153fcc-6843-4fd1-9d39-2c544b2fbba3.mp4
0: #C A motorcycle rides past C.
1: #C A motorcycle rides past C.
2: #C A car drives past C.
3: #C A white car drives past C
4: #C A motorcycle rides past C.
5: #C C drives past a car.
6: #C A white car drives past C
7: #C A motorcycle rides past C.
8: #C A black car drives past C
9: #C A white car drives past C
a2eb1eed-d930-431c-a3c2-e8bca2b1e22d.mp4
0: #O The man A holds the steering wheel with both hands.
1: #O The man A places his left hand on the car steering.
2: #O The man A holds the phone with his right hand.
3: #O The man B operates the car stereo with his right hand.
4: #O The man A drives the car with the steering wheel in his right hand.
5: #O The man X holds the steering wheel with his right hand.
6: #O The man X holds the phone with his right hand.
7: #O The man X holds the steering wheel with his left hand.
8: #O The man X operates the gear in the car with his right hand.
9: #C A motorcycle rides past C.
e466b8ee-f5a2-4bed-85d5-32b9f492629b.mp4
0: #C C holds a phone with both hands.
1: #C C drives past a car.
2: #O The woman X operates a phone with both hands.
3: #C C holds a phone with her left hand.
4: #C C drives the car with both hands.
5: #C C holds a phone with his right hand.
6: #C C drives a car with both hands. 
7: #C C drives the car with his left hand on the steering wheel
8: #O The woman B operates the phone with her right hand.
9: #C A motorcycle rides past C.
fb938133-badc-4ce5-a404-698d78f475c8.mp4
0: #O The woman B holds the steering wheel with both hands.
1: #C C drives the car with the steering wheel in his right hand.
2: #O A man D operates a phone in his left hand.
3: #O The woman A holds the phone with her left hand.
4: #O The man A holds a phone with his left hand.
5: #O The man B drives the car with both hands on the steering wheel
6: #C C drives past a man J.
7: #C C turns the steering wheel with his left hand.
8: #O A man J operates a phone with his left hand.
9: #C A motorcycle rides past C.
2ebfeb4c-6593-4077-b4b5-02669719a401.mp4
0: #C C looks around
1: #C C holds a steering wheel with his right hand. 
2: #C C drives the car with the steering wheel in both hands.
3: #C C drives the car with his right hand.
4: #C C drives the car
5: #C A blue car drives past C.
6: #C C drives the car with both hands.
7: #C A car drives past C.
8: #O A man X drives a car
9: #C C drives the car
2df8123e-8f18-4d89-a12b-5ef26f77b93e.mp4
0: #C A white car drives past C.
1: #C A motorcycle rides past C.
2: #C A man B rides a bicycle past C
3: #C A motorcycle rides past C.
4: #C A motorcycle rides past C.
5: #C A car drives past C.
6: #C A motorcycle rides past C.
7: #C A motorcycle rides past C.
8: #C A motorcycle rides past C.
9: #C A car drives past C.
df9bf80d-4800-4b8b-be08-b40c60a3231c.mp4
0: #C C drives past a truck on the road.
1: #C A motorcycle rides past C.
2: #C A motorcycle rides past C.
3: #C C drives past a motorcycle.
4: #C A motorcycle rides past C.
5: #C C drives past a motorcycle.
6: #O The man X operates the car stereo with his right hand.
7: #C A motorcycle rides past C.
8: #C A motorcycle rides past C.
9: #C C drives past the man X on a motorcycle.
10766d42-5b04-4a41-9623-3852906ff84e.mp4
0: #C A black car drives past C.
1: #C A motorcycle rides past C.
2: #C A motorcycle rides past C.
3: #C A car drives past C.
4: #C A motorcycle rides past C.
5: #C A yellow car drives past C.
6: #C C holds a steering wheel with both hands.
7: #C A yellow car drives past C.
8: #O The man A holds a phone in his right hand.
9: #C C drives the car with his right hand on the steering wheel.
f2fa7974-1451-436c-a52d-14cfcca4a54d.mp4
0: #C A car drives past C.
1: #C C turns the steering wheel with his right hand.
2: #C C turns the steering wheel with his right hand.
3: #C C turns to the side
4: #C C drives past a man G.
5: #C A motorcycle rides past C.
6: #C C turns the steering wheel with his right hand.
7: #C A yellow car drives past C.
8: #C A blue car drives past C.
9: #C C drives past a man R.
8a9073f9-5323-46bd-a571-1af07f714251.mp4
0: #C A white car drives past C
1: #C A red car drives past C
2: #C A white car drives past C
3: #C A white car drives past C
4: #C A car drives past C.
5: #C A white car drives past C
6: #C A black car drives past C
7: #C A white car drives past C
8: #C A car drives past C.
9: #C A car drives past C.
4b41cad6-8191-4480-a0f1-4de77a8942c2.mp4
0: #C C drives the car
1: #C C places his left hand on his right hand.
2: #C C drives the car with his left hand on the steering wheel.
3: #C C drives the car with the steering wheel in his left hand.
4: #C C turns the steering wheel with his left hand.
5: #C C turns right on the road
6: #C C turns right on the road.
7: #C C drives the car on the road with his right hand on the steering wheel
8: #C C drives the car with the steering wheel in his right hand.
9: #C C drives the car with the steering wheel in his left hand.
12e19cbc-112c-4004-b115-8d8440d68385.mp4
0: #C C turns left at a junction.
1: #C C drives the car with the steering wheel in his right hand.
2: #C C drives the car 
3: #C C turns the steering wheel with both hands.
4: #C C turns the steering wheel with his right hand.
5: #C C holds the steering wheel with his right hand.
6: #C C turns the steering wheel with his right hand.
7: #C C moves the steering wheel
8: #C C holds steering wheel with left hand
9: #C C drives the car with his left hand on the steering wheel.
ce77212b-72b9-467c-b2fa-a32b5f28b06c.mp4
0: #C C stops the car
1: #C C drives a car
2: #C C looks right
3: #C C drives the car
4: #C C looks around 
5: #C C drives the car
6: #C C turns left
7: #C C looks at the side mirror
8: #O Person A operates a phone
9: #C C drives a car
7869719a-ebf4-4065-a67d-e93a396071a7.mp4
0: #C A car drives past C.
1: #C A car drives past C.
2: #C A car drives past C.
3: #C C turns left on a road.
4: #C C drives the car
5: #C C drives the car on the road with both hands.
6: #C A motorcycle rides past C.
7: #C C drives a car on the road 
8: #C C turns to the side
9: #O A man X drives a car on the road.
ef3e747c-6756-45ff-aac1-80b5644bae6e.mp4
0: #C C looks around 
1: #C C drives the car with his left hand on the steering wheel.
2: #O A man Y drives the car
3: #O The man A places the phone in his right hand on the car seat
4: #C C places his left hand on his lap.
5: #O A man X drives the car
6: #C C drives the car
7: #C C turns a steering wheel with his right hand.
8: #O A man Z makes a gesture 
9: #O The man A holds the phone with both hands.
98249b55-f9ad-4388-85ed-5d7708c77fc7.mp4
0: #C C drives past a car.
1: #C A car drives past C.
2: #O The man B operates a phone with his right hand.
3: #C C drives past a yellow car.
4: #O The man A operates a phone with his right hand.
5: #C A motorcycle rides past C.
6: #C C drives past a motorcycle.
7: #O The man A operates the phone with his right hand.
8: #C A car drives past C.
9: #C A car drives past C.
aca6db60-e421-4cf1-86cb-91f65e7ac7f3.mp4
0: #C C drives past a car.
1: #O A yellow car drives past C.
2: #O A white car drives past C.
3: #C A white car drives past C.
4: #C A white car drives past C.
5: #C A black car drives past C.
6: #C A motorcycle rides past C.
7: #O A red car drives past C.
8: #C A truck drives past C.
9: #C A car drives past C.
8604edbc-26f8-4d89-8324-cfc6eca9798c.mp4
0: #O The man A holds the steering wheel with his right hand.
1: #C C drives the car with both hands on the steering wheel.
2: #O A man X holds a phone in a car with his right hand
3: #O A car drives past C.
4: #O A white car drives past C.
5: #O A yellow car drives past C.
6: #O The man X touches the car's steering wheel with his right hand
7: #O The man X touches his face with his left hand.
8: #O The man D operates the car's car control with his left hand
9: #O A man X drives the car with his right hand.
e097cc97-0357-4a85-8d58-bda4171e0764.mp4
0: #C C drives the car
1: #C C turns left on the road
2: #C C drives the car
3: #C C looks around the road
4: #C C turns left on the road 
5: #C C turns the steering wheel
6: #C C drives the car
7: #C C turns the steering wheel
8: #C C drives the car
9: #C C drives the car with both hands.
11b45298-9193-4153-90f5-5e37e22fbc72.mp4
0: #C C drives past a motorcycle.
1: #C A motorcycle rides past C.
2: #C A car drives past C.
3: #C C drives past a car.
4: #C A white car drives past C.
5: #C A motorcycle rides past C.
6: #C C drives past a yellow car.
7: #C C drives past a white car.
8: #C C drives past a motorcycle.
9: #C C holds a steering wheel with his right hand
cd5b2961-8aed-4805-90bb-8bf2d2798d0a.mp4
0: #C C looks around the road
1: #O A man X touches the chin with the right hand
2: #O Person B puts the phone on the pocket
3: #O Person B operates on his phone
4: #O man X uses the phone
5: #C C looks at man X
6: #C C looks at man X
7: #O Person A holds his phone
8: #O A man X touches his face with his right hand.
9: #O The man X places his right hand on his face.
f671f270-b1b8-42b1-a059-dd218631e4ef.mp4
0: #O a person M uses a phone
1: #O A man A operates a phone with both hands.
2: #C A motorcycle rides past C.
3: #O A man Y uses a phone 
4: #O man X operates a phone
5: #C C looks around
6: #O The man B operates the phone with both hands.
7: #C C looks at the road 
8: #C C looks around
9: #O A car drives past C.
60a02721-9045-47bf-bf02-f492589e3991.mp4
0: #C C drives car on the road
1: #O The man X holds the phone with his left hand.
2: #O man X uses his phone
3: #O The man X drives the car with both hands.
4: #O The man X holds the phone with both hands.
5: #O The man B operates the phone with his left hand.
6: #C A white car drives past C.
7: #C C drives past a motorcycle.
8: #O a person T uses the phone
9: #O The man A operates the phone with his right hand.
5d18d116-729a-4006-ade2-a31a2d73e160.mp4
0: #C C drives the car with both hands.
1: #C C holds the steering wheel with his right hand.
2: #C C drives the car with the steering wheel in both hands.
3: #C C drives the car with his left hand. 
4: #C C moves his left hand on the steering wheel.
5: #C C holds a steering wheel with his left hand.
6: #C C drives the car with the steering wheel in his right hand
7: #C C holds a steering wheel with his left hand.
8: #C C drives the car with his left hand.
9: #C C drives the car with the steering wheel in both hands.
f5115be8-a70b-4971-aef3-c640d5edbc86.mp4
0: #C A car drives past C.
1: #C C drives the car with his right hand on the steering wheel
2: #C C drives past a yellow car.
3: #C C drives the car with the steering wheel in her right hand
4: #C C drives past a yellow car.
5: #C C drives past a car.
6: #O A yellow car drives past the car.
7: #C A car drives past C.
8: #C A motorcycle rides past C.
9: #C C turns the steering wheel with his right hand.
a21eb68b-a25e-4b7e-b0b2-97219b794008.mp4
0: #O The man X turns the steering wheel with both hands.
1: #O man Y drives car 
2: #C A yellow car drives past C.
3: #C A white car drives past C.
4: #O The man Z operates the car's wiper switch with his right hand
5: #O A man E drives the car with the steering wheel in both hands.
6: #C A white car drives past C.
7: #O man Y holds a car steering wheel 
8: #C C turns towards the man Y.
9: #C A car drives past C.
ef6a8ead-bd09-4b9f-a00f-f5bc0e9341ae.mp4
0: #C A car drives past C.
1: #O The man A holds a phone with his left hand
2: #C A car drives past C.
3: #C A motorcycle rides past C.
4: #C C drives past a car.
5: #C A black car drives past C.
6: #C A car drives past C.
7: #C A car drives past C.
8: #C C drives past a car.
9: #C C drives past a car.
8e816c0d-7ca3-4bc6-833f-33d8338d745f.mp4
0: #C A white car drives past C
1: #C A car drives past C.
2: #C A car drives past C.
3: #C A white car drives past C
4: #C A black car drives past C
5: #C A white car drives past C
6: #C A white car drives past C
7: #C A motorcycle rides past C.
8: #C A white car drives past C
9: #C A white car drives past C
ecb55c22-3d6c-40f6-a791-51c43330f0d0.mp4
0: #O The woman X operates the phone with both hands.
1: #O The woman X operates the phone with her right hand.
2: #C C drives past a white car.
3: #C C drives the car with the steering in his right hand.
4: #O A black car drives past C.
5: #C C holds the steering wheel with his right hand.
6: #C C drives the car with the steering wheel in his left hand
7: #C C turns to the side
8: #C A red car drives past C.
9: #C C drives the car with his right hand.
2a4a188f-2d7f-4b90-8d74-abb8aec9ea93.mp4
0: #C C drives the car with his right hand on the steering wheel.
1: #C C drives the car with his right hand.
2: #C C drives the car with the steering wheel in both hands.
3: #C C drives the car with his left hand on the steering wheel.
4: #C C drives the car with his right hand on the steering wheel.
5: #C C stops the car at a traffic light with his right hand on the steering wheel.
6: #C C drives the car with the steering wheel in his hands.
7: #C C drives the car with the steering wheel in both hands.
8: #C C drives the car with his right hand on the steering wheel.
9: #C C drives the car
e9475f55-7bf9-4181-b2b1-c342a332f77d.mp4
0: #C C drives the car with his right hand.
1: #C C drives the car with his right hand on the steering wheel.
2: #O A woman A operates a phone with her right hand.
3: #C C holds the steering wheel with both hands.
4: #C C drives the car with both hands.
5: #C C drives the car
6: #C C drives the car 
7: #O the woman A operates the phone with both hands.
8: #C C drives the car with both hands.
9: #C C turns a steering wheel with both hands.
bfebbca7-5fb9-43e1-bb8a-c9f4d5900d52.mp4
0: #C C holds the steering wheel with his right hand.
1: #O the woman A operates the car gear with her right hand.
2: #O A white car drives past C.
3: #C C turns a steering wheel with her right hand.
4: #O The man D operates the phone with his right hand.
5: #O The man X operates a phone with both hands.
6: #C C turns car steering
7: #O The woman X operates a phone with both hands.
8: #O The man X operates the car radio with his left hand.
9: #C C drives past a car.
98d36605-75b8-4011-9ebd-0a55e6f9b30a.mp4
1: #C C drives the car 
2: #C C drives past a bike on the road.
3: #C C drives past a truck.
4: #C A white car drives past C.
5: #C C drives past a bus.
6: #C A motorcycle rides past C.
7: #C C drives past a car.
8: #C C drives past a car.
9: #C C drives past a blue car.
e8802364-e16a-41b8-ac09-3617d9a4f5cb.mp4
0: #C C drives a car 
1: #O A man Z looks at the phone 
2: #C C looks around 
3: #C C looks at the side mirror 
4: #O A man Y holds a phone with both hands
5: #C C turns right on the road
6: #C C looks at the road
7: #O The man X operates the phone with his left hand
8: #O A man Z operates a phone
9: #O A man Z drives the car
19cbaeab-602e-4b1c-a2f2-1f9e3311832e.mp4
0: #C A white car drives past C.
1: #C A black car drives past C.
2: #C C drives past a white truck.
3: #C A car drives past C.
4: #C A car drives past C.
5: #O The man A operates the phone in his left hand with his right hand
6: #C A white car drives past C.
7: #O The man B operates the phone with both hands.
8: #C A motorcycle rides past C.
9: #C A motorcycle rides past C.
badbb962-3388-4c45-8e02-31cec46fdfb5.mp4
0: #C C drives the car with his left hand.
1: #O A woman Y holds a phone with her right hand.
2: #O The woman X drives the car with both hands on the steering wheel
3: #O The woman A removes her right hand from the steering wheel.
4: #O The man B turns the steering wheel with his hands.
5: #C C drives past a white car.
6: #C C drives the car with the steering wheel in his right hand.
7: #O A woman D operates a phone with both hands.
8: #O The woman B drives past the car.
9: #O The man X holds the steering wheel with his right hand.
32a68148-8c4b-41be-8819-a22213257a35.mp4
0: #C A yellow car drives past C.
1: #C C drives the car with the steering wheel in both hands
2: #O A man K drives a car
3: #C C drives the car with his right hand.
4: #C C drives the car with his right hand.
5: #O A man Y drives the car
6: #C A black car drives past C.
7: #O A man Y drives the car
8: #C C looks around
9: #C C drives a car
e2a3675c-8df1-4efe-b7e3-cc9f217ce581.mp4
0: #C A car drives past C.
1: #C A black car drives past C
2: #C A white car drives past C
3: #C A white truck drives past C
4: #C C drives past a white car
5: #C C drives past a white car
6: #C C drives past a white truck
7: #C A motorcycle drives past C.
8: #C A white truck drives past C
9: #C C drives past a grey car
c21d73e1-a718-4be4-9c9f-875e04c678ab.mp4
0: #C A car drives past C.
1: #C C drives past a white car.
2: #C A car drives past C.
3: #C A car drives past C.
4: #C A car drives past C.
5: #C A car drives past C.
6: #C A car drives past C.
7: #C A car drives past C.
8: #C A car drives past C.
9: #C A car drives past C.
10455a33-71cc-4c95-b764-b78875594aaf.mp4
0: #C C turns car steering
1: #C C drives the car on the road with his right hand on the steering wheel
2: #C C places his right hand on the steering wheel.
3: #C C turns the steering wheel with his left hand.
4: #C C drives past a white car.
5: #C C drives the car on the road with his left hand on the steering wheel
6: #C C drives the car with his right hand on the steering wheel.
7: #C C drives past a white car.
8: #C C drives the car on the road.
9: #C C drives the car with his right hand.
735cf3fb-039f-4613-9d9c-3391551e0ecf.mp4
0: #C C drives past a white car.
1: #C a motorcycle drives past C.
2: #C A motorcycle rides past C.
3: #C A motorcycle rides past C.
4: #C A motorcycle rides past C.
5: #C A white car drives past C.
6: #C A man A rides past C.
7: #O A man X drives past the car
8: #C A motorcycle rides past C.
9: #C A black car drives past C.
645b86d4-fb1d-4500-a547-ff1a63942ca3.mp4
0: #C C drives the car
1: #C C drives the car 
2: #C C holds steering wheel with left hand
3: #C C drives the car
4: #C C drives the car with both hands.
5: #C C turns right on the road.
6: #C C holds a steering wheel with his left hand.
7: #C A white car drives past C.
8: #C C holds the steering wheel with his right hand.
9: #C C drives the car on the road
13de6440-0ef9-4632-9dbe-f49141cb3257.mp4
0: #C A motorcycle drives past C.
1: #C C drives past a yellow car.
2: #C C drives past a bike on the road.
3: #C C drives past a yellow car.
4: #C C drives past a red car.
5: #C C drives past a bike.
6: #C C drives past a man A on a bicycle on the road
7: #C C drives past a car.
8: #C A motorcycle rides past C.
9: #C C drives past a yellow car.
ec84331b-1f4c-4e50-ad45-9b314b3f5819.mp4
0: #C A black car drives past C
1: #C A blue car drives past C
2: #C A car drives past C.
3: #C A white car drives past C
4: #C C drives past a car.
5: #C A white car drives past C
6: #C A car drives past C.
7: #C A white car drives past C
8: #C A black car drives past C
9: #C A white car drives past C
88d9e3f9-2e06-4305-9a3f-07b59df248f5.mp4
0: #C C turns the steering wheel with both hands.
1: #C A motorcycle rides past C.
2: #C C drives past a truck.
3: #C C drives past a man T.
4: #C C drives past a car.
5: #C C drives the car with both hands on the steering wheel.
6: #C C drives past a bike.
7: #C A motorcycle rides past C.
8: #C C drives past a yellow car.
9: #C C drives past a bicycle.
79539643-6a3f-4c90-94d0-53a9135c5b43.mp4
0: #C C drives the car on the road with the steering wheel in his right hand
1: #C C looks at the car
2: #C C looks around
3: #C C drives the car with the steering wheel in his right hand.
4: #C C holds a car steering with his right hand.
5: #O The woman Y operates the phone with her right hand.
6: #C C looks around
7: #C C moves the hand
8: #C C moves the right hand
9: #C C moves the left hand 
56806da1-66c3-4484-99dc-68b93b414ed0.mp4
0: #O a person M uses the phone
1: #O A man Y operates a phone with his right hand
2: #O A Man H operates a phone with both hands 
3: #C C looks around
4: #C C looks around 
5: #C C looks at the left side mirror
6: #O A man X operates a phone with both hands.
7: #O person Y moves the car 
8: #O The man A operates the phone in his left hand
9: #C C looks around
179b16d3-1521-44bd-a15a-5eeef27719dd.mp4
0: #C A motorcycle rides past C.
1: #C A motorcycle rides past C.
2: #C A motorcycle rides past C.
3: #C A motorcycle rides past C.
4: #C A motorcycle rides past C.
5: #C A motorcycle rides past C.
6: #C A motorcycle rides past C.
7: #C C drives past a motorcycle.
8: #C A black car drives past C
9: #C A motorcycle rides past C.
d1ac616a-294c-4eab-a95a-e115df0e426e.mp4
0: #C C turns right on the road.
1: #C C holds the steering wheel with his left hand.
2: #C C drives the car.
3: #O Person A drives the car
4: #C C drives the car on the road
5: #C C moves the steering wheel.
6: #O Person A turns to the left
7: #C C turns to the side
8: #O A man X holds a phone
9: #C C moves the steering wheel with his right hand.
99900c7f-24f0-4562-b707-a9021968e4b8.mp4
0: #C C drives the car with his right hand.
1: #O A man X uses phone 
2: #O The man A operates the phone with his right hand.
3: #O A man Y operates a phone 
4: #C A car drives past C.
5: #O A man P holds the phone with his right hand.
6: #C C drives the car with his right hand.
7: #O A man A operates a phone with both hands.
8: #C C drives the car with the steering wheel in his left hand
9: #C C drives the car with the car steering in his right hand
b1589ed8-b74b-4d6c-a6e4-adb8479a1a15.mp4
0: #O person B holds a phone
1: #C C looks at the side mirror
2: #C C puts his right hand on his face.
3: #C C drives the car
4: #C C looks at the road 
5: #C C turns right
6: #O A man X drives the car 
7: #O person Y holds a car seat.
8: #C C holds a steering wheel with both hands 
9: #C C moves his right hand
264243d8-8a00-4681-819d-7abe27975a0e.mp4
0: #O A man X operates a phone with his right hand.
1: #C C drives the car with his left hand. 
2: #C A motorcycle rides past C.
3: #C A motorcycle rides past C.
4: #O A man Y operates a phone in a car with his right hand
5: #C C holds the phone with both hands.
6: #O The man A holds the phone with his right hand.
7: #O The man A operates the phone with his right hand.
8: #C C drives past a blue car.
9: #C A white car drives past C.
084e6321-a61b-4f51-90b8-58cf852788e2.mp4
0: #C C drives car on the road
1: #C C looks around
2: #C C drives a car 
3: #C C drives the car
4: #C C turns right
5: #C C turns right
6: #C C looks around
7: #C C turns left on the road
8: #C C drives the car on the road
9: #C C drives the car
1d4e2822-0930-4207-9fc2-0f8e6d9ee1be.mp4
1: #C A motorcycle rides past C.
2: #O The man Y operates the phone with both hands
3: #C A white car drives past C.
4: #C A motorcycle drives past C.
5: #C C drives past a white car.
6: #C A motorcycle drives past C.
7: #O A man Y uses the phone
8: #C A motorcycle rides past C.
9: #C C drives past a car.
e1bc67be-2f96-469b-bafb-93798ec6d78b.mp4
0: #O The man A operates the phone with his right hand.
1: #O The man X drives past a car.
2: #O The man B holds a phone with his left hand.
3: #C C looks around
4: #O The man A holds a phone with his right hand.
5: #O The man X operates the phone with his right hand.
6: #O A man Y drives the car
7: #O The man X operates a gear shift on the car dashboard with his right hand
8: #C C turns towards the man B.
9: #O A white car drives past C.
2fbce1ed-8fba-44ab-a425-0893056ebb3f.mp4
0: #O A man X operates a phone with both hands.
1: #O The man A operates the phone with his right hand
2: #O A man Y operates a phone
3: #O The man X operates the phone with both hands.
4: #C C looks around 
5: #O The man X operates the phone with both hands.
6: #C A black car drives past C.
7: #C C looks around
8: #C C looks around the car
9: #C C looks around
532e692a-a8fa-4f2d-969f-c6f76d8f5c23.mp4
0: #C C drives the car with both hands.
1: #C C drives the car with the steering wheel in both hands.
2: #C C drives the car with his left hand on the steering wheel.
3: #C C drives the car with the steering wheel in his right hand.
4: #C C turns left on the road.
5: #C C turns the steering wheel with his right hand.
6: #C C turns right on the road.
7: #C C turns the steering wheel with his left hand.
8: #C C drives the car with the steering wheel in his left hand.
9: #C C places his left hand on his lap.
39175ac5-2bcf-4703-8a10-235c8c38f9b2.mp4
0: #C A motorcycle rides past C.
1: #C A black car drives past C.
2: #C A motorcycle rides past C.
3: #C A motorcycle rides past C.
4: #C C drives past a bicycle.
5: #C C drives past a motorcycle.
6: #C A black car drives past C.
7: #C C drives the car with both hands.
8: #C C drives past a motorcycle.
9: #C C holds the steering wheel with his right hand
64bb06f3-590c-441f-bbf1-4393e9055282.mp4
0: #O The man A operates the phone in his right hand with his left hand
1: #O The man A operates the phone with his right hand.
2: #O The man A holds the phone with his left hand.
3: #O A man Q operates a phone with both hands.
4: #O A man B holds a phone with his right hand. 
5: #O The man A operates the phone with both hands.
6: #O The man A holds the steering wheel with his right hand.
7: #C A blue car drives past C.
8: #O The man D operates a phone with both hands.
9: #O The man A presses the car stereo with his right hand.
c4731395-705f-4fba-a8c1-84e6a190d2d2.mp4
0: #C C drives past a man X.
1: #C A motorcycle rides past C.
2: #C C drives a car
3: #C C drives past a black car.
4: #C C drives past a man Y.
5: #C C drives past a white car.
6: #C C drives past a car.
7: #C C drives past a yellow car.
8: #C C drives past a black car.
9: #C C drives past a car.
ff1bcf2f-16f5-471f-8ddf-02d6b8762b35.mp4
0: #C A white car drives past C.
1: #C C drives the car on the road with his right hand on the steering wheel
2: #C C drives the car with his right hand on a steering wheel.
3: #O The man B holds the phone in his right hand. 
4: #C A car drives past C. 
5: #C A white car drives past C.
6: #O The man Y drinks from the glass of beer with his right hand.
7: #C A white car drives past C.
8: #C C drives the car with both hands.
9: #C A white car drives past C.
ad9d4e70-6cd5-44f3-86d8-5441c6fdc8c3.mp4
0: #O The man Y operates the phone with his right hand.
1: #O A man Y drives the car
2: #C C looks around
3: #C C looks on the road
4: #C C looks around
5: #C C drives a car
6: #C C looks around 
7: #O A man B  drives the car on the road.
8: #C C turns the car steering wheel to the right with his right hand
9: #C C turns left on the road
1cc12a18-ee5f-4d22-a5fb-23c6a4142a13.mp4
0: #C C drives the car.
1: #C C looks around the road
2: #C C drives the car.
3: #O person X operates a phone
4: #C C drives a car on the road
5: #C C looks forward
6: #C C drives a car
7: #O person X drives a car
8: #C C drives the car
9: #C C looks around 
2d96dd89-742b-4d16-ae0d-e4a7877ad18b.mp4
0: #O A man Y operates a phone with his right hand
1: #C C removes his right hand from the steering wheel.
2: #C C holds the steering wheel with his right hand.
3: #O A woman A operates the phone with her left hand
4: #C C turns the steering wheel with both hands.
5: #O The woman Y operates a phone with her right hand
6: #O The man A operates the phone with both hands.
7: #O A woman X operates a phone with her right hand
8: #C C removes his right hand from the steering wheel.
9: #C C turns the steering wheel with his right hand.
1956879c-a3e6-4ae0-a2b4-2bb771c8eeab.mp4
0: #C C looks at the road
1: #O person Z operates a phone
2: #O a person M uses the phone
3: #O person A operates the phone
4: #C C looks around
5: #O The man X operates a phone with both hands.
6: #C C looks around
7: #C C looks around
8: #C C looks around
9: #O A woman D operates a phone with both hands.
155ab506-98c9-4c8b-bc9f-d7fdd13049af.mp4
0: #C C drives car on the road
1: #C C drives car on the road
2: #C C drives the car 
3: #C C drives car on the road
4: #C C drives car on the road
5: #C C drives car on the road
6: #C C looks around
7: #C C looks around
8: #C C looks around
9: #C C looks around
d1711cc9-6e02-431c-a259-ce241b39d135.mp4
0: #O The man X operates the phone with both hands.
1: #O The man Y holds the phone with both hands.
2: #C C drives the car with the steering in his right hand
3: #O man X uses the phone 
4: #C C turns towards a man X.
5: #O The man X operates the phone with both hands.
6: #O person X operates the phone
7: #O A man X operates a phone with his right hand.
8: #O The man B operates the phone with his right hand.
9: #O The man X operates the phone with his right hand.
893383b0-75a6-4bbb-9486-95ee0809c6db.mp4
0: #O A man X drives the car
1: #C C drives the car on the road with both hands
2: #C C drives the car 
3: #O person Y drives the car.
4: #C C turns the steering wheel with his right hand.
5: #O A man X drives the car
6: #C C drives the car
7: #C C drives the car on the road
8: #C C looks around 
9: #C C moves hand on the steering wheel
15356382-e28f-409a-b4fb-22054e3b47af.mp4
0: #C C drives the car with the steering wheel in his right hand
1: #C C drives the car with his right hand.
2: #C C drives the car with his right hand on the steering wheel
3: #C C drives the car with his right hand on the steering wheel
4: #C C drives past a car.
5: #C A white car drives past C.
6: #C C drives the car with his left hand on the steering wheel
7: #C A black car drives past C.
8: #C A car drives past C.
9: #C C turns the steering wheel with his right hand.
a65828a5-a4bf-4a0a-a340-ffa910a3752f.mp4
0: #C C drives the car on the road with the steering wheel in his left hand
1: #C A car drives past C.
2: #O A Man Y operates phone with right hand 
3: #C C drives car on the road
4: #C C looks at the road
5: #O A Woman M uses phone 
6: #O A man Z operates a phone with both hands.
7: #C C looks around
8: #O A Man X holds phone with the left hand 
9: #O A woman X operates a phone with both hands.
29ca64a8-af0e-44fb-a801-c9c490108b2d.mp4
0: #C C moves the gear lever
1: #C C drives the car with both hands on the steering wheel.
2: #C A car drives past C.
3: #C C drives past a car.
4: #C C drives past a red car.
5: #C C holds the steering wheel with his right hand.
6: #C C drives the car with the steering wheel in his left hand
7: #C C holds the gearshift with his left hand.
8: #C C drives past the white truck.
9: #C A white car drives past C.
5c8b745c-64a5-4514-9937-ae930e1802c6.mp4	
cccccccccccccccccccccccccccccccccccccccccccccccccccccc0: #C C drives past a bike.
1: #O The man X operates a phone with both hands.
2: #C C drives past a motorcycle.
3: #C C drives past a car.
4: #C C drives past a motorcycle.
5: #C C turns the car on 
6: #C C drives past a yellow tricycle.
7: #C C drives past a motorcycle.
8: #C C drives past a truck.
9: #C C drives the car on the road with his right hand
ec4e21cb-8d1f-4f6c-b483-52928f7ba558.mp4
0: #O a person P operates a phone
1: #O person X operates a mobile phone.
2: #O person X operates a phone. 
3: #C C looks at a phone
4: #O person X uses phone.
5: #O person A operates a phone
6: #O A man X operates a phone in the car with his right hand
7: #O person B operates a phone.
8: #C C looks around the road
9: #O person X holds a phone.
23cf03a8-0216-4c07-aa12-31bf6997a396.mp4
0: #O The man B holds the phone with his left hand.
1: #C A yellow car drives past C.
2: #C A motorcycle rides past C.
3: #O The man A presses the phone with both hands.
4: #C C drives a car
5: #O A man T holds a phone with his left hand.
6: #C C drives the car with his right hand.
7: #O The man A presses a phone in his right hand with his left hand
8: #O A man B operates a phone with his left hand.
9: #C C drives the car with both hands.
21f6edd3-d0bd-4155-9b9b-0419e41ba277.mp4
0: #O A woman A operates a phone with her right hand.
1: #C C drives past a man P.
2: #C A black motorcycle rides past C.
3: #O The man D operates a phone with his right hand.
4: #O A man A operates a phone with his left hand.
5: #C A white car drives past C.
6: #O A man B rides a bicycle on the road.
7: #O The woman B operates a phone with her right hand.
8: #O The woman A operates the phone with both hands.
9: #O The woman X operates the phone with her right hand.
63309313-b8fb-4150-ab4c-6375a23b9667.mp4
0: #O A man K operates a phone with his right hand.
1: #C A white car drives past C.
2: #O A man X operates a phone with his right hand.
3: #O A man D drives a car.
4: #O The man D operates the phone with his right hand.
5: #C C drives past a man H on a motorcycle.
6: #C A motorcycle rides past C.
7: #O The man A drives the car with the steering wheel in his right hand
8: #C A car drives past C.
9: #C C looks at the side mirror
53559247-6d13-4011-9a26-6cfefbe4480b.mp4
0: #C A white car drives past C
1: #C A white car drives past C
2: #C A black car drives past C
3: #C A white car drives past C
4: #C A white car drives past C
5: #C A black car drives past C
6: #C A black car drives past C
7: #C C drives past a truck.
8: #C A car drives past C.
9: #C A white car drives past C
df8726b3-8174-454d-b866-37189f5e17d3.mp4
0: #C C looks at the side mirror 
1: #C C moves his right hand on the steering wheel.
2: #C C drives past a car.
3: #O The man X operates the gear shift with his right hand
4: #O The man X holds the steering wheel with his right hand
5: #C C moves the car gear lever with his left hand.
6: #C C turns the steering wheel with his right hand.
7: #C C looks around
8: #O The man A holds a phone with his right hand.
9: #O A yellow car drives past C.
cab5ad44-efcc-40fe-bcbe-7b35dc33d6cc.mp4
0: #C C drives past a bike.
1: #C A motorcycle rides past C.
2: #C A motorcycle rides past C.
3: #C A car drives past C.
4: #C A motorcycle rides past C.
5: #C A man A rides past C.
6: #C A motorcycle rides past C.
7: #C A motorcycle drives past C.
8: #C C drives past a motorcycle.
9: #C C drives past a motorcycle.
d9e622d2-b0c2-4afa-9631-7f59f3d68049.mp4
0: #O The woman X holds the phone with both hands.
1: #C C drives past a motorcycle.
2: #C C drives past a man R.
3: #C C turns the car steering with his left hand.
4: #C C drives a car
5: #O A Woman M uses phone 
6: #C C turns left on a road.
7: #C A motorcycle rides past C.
8: #C C drives the car with his left hand on the steering wheel
9: #C C turns towards a woman X.
ccc5bad4-8b34-4d59-923d-10ca55a043b1.mp4
0: #O A man X places his right hand on a steering wheel
1: #C C looks around
2: #O A man X drives a car
3: #C C looks around the road
4: #C C looks around
5: #O A man X drives the car
6: #C C looks at the side mirror
7: #O A man X operates a phone
8: #O A man X holds a phone in his left hand.
9: #O Person A operates a phone
a13fe4ae-c414-4fa4-a00c-2a505e0365e0.mp4
0: #C C turns left
1: #C C looks around
2: #C C drives the car
3: #C C drives a car
4: #C C looks around
5: #C C drives the car with his right hand on the steering wheel.
6: #C C turns left on the road.
7: #C C drives a car
8: #O A man A operates a phone in a car with his left hand
9: #C C drives a car
a49cb770-d26d-4f52-967c-f84e692a05b9.mp4
0: #C C looks around
1: #C C holds a phone
2: #C C turns left on the
3: #C C moves the left hand
4: #C C looks around 
5: #C C drives a car
6: #O Person B operates a phone
7: #C C stops the car
8: #C C looks around
9: #O Person A operates a phone
830af93b-5cdb-43ba-8a54-d3dbe5b78fa2.mp4
0: #C C drives the car with both hands.
1: #C C drives the car with the steering wheel in his right hand
2: #C C holds a steering wheel with her right hand.
3: #C C holds a steering wheel with his right hand.
4: #C C drives the car on the road with his left hand.
5: #C C drives a car
6: #C A car drives past C.
7: #C A man J rides a bicycle on the road past C.
8: #C C drives the car with both hands. 
9: #C C drives the car with his right hand.
3c3d3a95-cedc-4b7d-a180-0bf6147e420c.mp4
0: #C C drives past a white car.
1: #C C turns the steering wheel with both hands.
2: #C C drives past a motorcycle.
3: #C C holds the steering wheel with his right hand.
4: #C C drives the car with his right hand.
5: #O A man B operates a phone with both hands.
6: #C C drives the car with his right hand on the steering wheel
7: #C C drives the car on the road with both hands.
8: #C C drives the car with both hands.
9: #C C turns the steering wheel with his right hand.
4fb5a9df-1fb7-4da7-bbba-1c1d7e4c0739.mp4
0: #C A car drives past C.
1: #O A man B drives the car.
2: #C A motorcycle rides past C.
3: #O The man A operates a phone with his right hand.
4: #C C drives the car
5: #C A motorcycle rides past C.
6: #C A motorcycle rides past C.
7: #C A black car drives past C.
8: #C C drives past a car.
9: #C C drives car on the road
150acb24-9c8f-4e99-8e35-eb5cc44bc770.mp4
0: #C C turns right
1: #C C drives a car
2: #C C turns left at a junction.
3: #C C turns to the side
4: #C C turns right on the road.
5: #C C drives a car 
6: #C C turns the steering wheel with his left hand
7: #C C turns a steering wheel with his right hand
8: #C C turns the car steering with his right hand
9: #C C drives a car
c945ac8d-485f-4035-851d-6cbbaf8b1fd1.mp4
0: #C A black car drives past C
1: #C A white car drives past C
2: #C A white car drives past C
3: #C C drives past a blue car
4: #C A black car drives past C
5: #C A motorcycle rides past C.
6: #C C drives past a car.
7: #C A white car drives past C
8: #C A car drives past C.
9: #C A white car drives past C
7d67b96f-c813-4bc5-8625-bf56e947c14e.mp4
0: #O The woman A operates the phone with both hands.
1: #C C drives along the road.
2: #C A car drives past C.
3: #C C looks around
4: #O The woman A operates the phone with her right hand.
5: #O The man A operates the phone with both hands.
6: #O The woman D operates the phone in her right hand.
7: #C A black car drives past C.
8: #C A white car drives past C.
9: #O A woman Y operates a phone with both hands.
0445e940-260f-49f4-84cf-3fe61c6935d8.mp4
0: #C C looks around
1: #C C turns to the side
2: #C C drives the car on the road with the steering wheel in his hands
3: #C C looks around the road
4: #C C drives the car on the road with both hands.
5: #C C drives the car on the road with both hands.
6: #C C holds the steering wheel with his left hand.
7: #C C turns right
8: #C C turns right 
9: #O The man X holds the steering wheel with his right hand.
e7d136de-5e5d-4d83-865b-97e262ba3e91.mp4
0: #O person A uses the phone
1: #C C looks around
2: #O person A holds a phone with the right hand
3: #O Person B operates a phone
4: #O person B holds the phone
5: #C C looks around
6: #C C looks at the road
7: #O person A operates the phone
8: #C C looks at the road 
9: #O person A scrolls on a phone
c2d6d882-7679-474e-9997-8ebbe681d1fa.mp4
0: #C A car drives past C.
1: #C C drives the car with both hands.
2: #C A white car drives past C.
3: #C A car drives past C.
4: #C C drives past a blue car.
5: #C C drives the car
6: #C C drives the car on the road with his left hand
7: #C C holds the steering wheel with both hands.
8: #C A car drives past C.
9: #C A car drives past C.
01dc0323-cfe3-4558-96e2-7818500368fe.mp4
0: #C C drives the car with his right hand.
1: #C C drives the car with his right hand on the steering wheel.
2: #C C drives the car with his left hand.
3: #C C drives the car with the steering wheel in his right hand.
4: #C C holds the steering wheel with his left hand.
5: #C C drives the car with both hands on the steering wheel.
6: #C C holds the steering wheel with both hands.
7: #C C turns the steering wheel with his right hand.
8: #C C drives the car with both hands.
9: #C C drives the car with both hands.'''



# Split the input data by lines
lines = data.splitlines()

current_video = None
sentences = []
video_responses = {}
# Loop through the lines of the input data
for line in lines:
    line = line.strip()  # Remove leading/trailing whitespaces
    if line.endswith(".mp4"):  # If the line is a video filename
        if current_video and sentences:
            # Store the previous video and its corresponding sentences in the dictionary
            video_responses[current_video] = sentences
        
        # Start a new video entry
        current_video = line
        sentences = []  # Reset the sentences list for the new video
    elif line and current_video:  # If the line is a sentence and there's a current video
        sentences.append(line)  # Add the sentence to the list

# Store the last video and its corresponding sentences
if current_video and sentences:
    video_responses[current_video] = sentences

# classes = {0:'right turn', 1:'right lane change', 2:'left turn', 3:'left lane change', 4:'end action'}
# count = {'end action':0, 'left lane change':0, 'left turn':0, 'right lane change':0, 'right turn':0 }
# total = {'end action':0, 'left lane change':0, 'left turn':0, 'right lane change':0, 'right turn':0 }

classes = {0:'straight', 1:'slow down', 2:'left turn', 3:'left lane change', 4:'right turn', 5:'right lane change', 6:'u turn'}
count = {'straight':0, 'slow down':0, 'left turn':0, 'left lane change':0, 'right turn':0, 'right lane change':0, 'u turn':0}
total = {'straight':0, 'slow down':0, 'left turn':0, 'left lane change':0, 'right turn':0, 'right lane change':0, 'u turn':0}


# # Load model and tokenizer
# model_name = "meta-llama/Meta-Llama-3.1-8B"  # Replace with the correct model name
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# tokenizer.pad_token = tokenizer.eos_token
# # Load the Llama model for sequence classification (assuming we use a classification head)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7)  # num_labels=5 for multiclass classification
# # model = AutoModelForCausalLM.from_pretrained(model_name)

# # Move model to GPU if available
# device = torch.device("cuda" if False else "cpu")
# model.to(device)

# import pandas as pd
# import re
# from collections import defaultdict

# gd = pd.read_csv('/scratch/sai/train.csv')
# # gd['filename']=gd['filename'].apply(lambda x: x.split('/')[-1])
# d = defaultdict(list)


# for filename, reponse in video_responses.items():
    
#     new_rep = []
#     for res in reponse:
#         if ".txt" in res:
#             continue
#         else:  
#             res = re.sub(r'\d+: #C', '', res)
#             new_rep.append(res)


#     # Define the input prompt text
#     input_text = '#Caption - '.join(new_rep) + ''' Given the narration of the driving video marked as #Caption classify the manuver into these  classes: target_labels = {
#         "straight": 0,
#         "slow down": 1,
#         "left turn": 2,
#         "left lane change": 3,
#         "right turn":4,
#         "right lane change":5,
#         "u turn": 6
#     }'''

#     # print(input_text)
#     #Given the narration of the driving video marked as #Caption classify the manuver i have 7 clsses - 
#     #Caption - Input_text (genrated by VLM)
#     #

#     # Define different target labels (class indices)
#     target_labels = {
#         "straight": 0,
#         "slow down": 1,
#         "left turn": 2,
#         "left lane change": 3,
#         "right turn":4,
#         "right lane change":5,
#         "u turn": 6
#     }
#         # Calculate padding to match input length and pad the target text if necessary


#     # target_labels = {
#     #     "straight",
#     #     "slow down",
#     #     "left turn",
#     #     "left lane change",
#     #     "right turn",
#     #     "right lane change",
#     #     "u turn"
#     # }
#     # Tokenize input text
#     inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

#     predicted = {}
#     # Iterate through each possible target label and compute the loss
#     for target_label_text,target_label_idx  in target_labels.items():
#         # Tokenize the target label text (this is the ground truth for the task)
#         labels = torch.tensor([target_label_idx]).to(device)  # Convert label text to corresponding index
#         # labels = tokenizer(target_label_text, return_tensors='pt').to(device)

#         # input_length = inputs['input_ids'].shape[1]  # Length of input text
#         # target_length = labels['input_ids'].shape[1]  # Length of target label text

#         # if input_length > target_length:
#         #     # Pad the target to match the input length
#         #     labels['input_ids'] = tokenizer.pad(
#         #         labels,
#         #         padding="max_length",
#         #         max_length=input_length,
#         #         return_tensors="pt"
#         #     )["input_ids"]

        
        
#         # Forward pass through the model
#         outputs = model(input_ids=inputs["input_ids"], labels=labels)
#         import pdb;pdb.set_trace()
      
#         # The model output includes the logits and the loss (computed automatically)
#         logits = outputs.logits
#         loss = outputs.loss  # Cross-entropy loss automatically calculated

#         # # Print the loss value for the current label
#         # print(f"Target Label: {target_label_text} | Loss: {loss.item()}")

#         # You can also compute the predicted class by finding the maximum logit value
#         predicted_class = torch.argmax(logits, dim=1).item()
#         predicted[classes[predicted_class]] = loss.item()
#         # print(f"Predicted Class: {predicted_class} (Index) | Target Class: {target_label_idx}")

#     answer = min(predicted.items(), key=lambda x:x[1])[0]
    
    
#     if any(gd['name']==filename):
#         flag = False
#         ans = classes[gd[gd['name']==filename]['class'].to_list()[0]]
#         if answer == ans:
#             count[ans] += 1
#             flag = True
#         else:
#             print(answer , ans)
#         total[ans] += 1
#         d['filename'].append(filename) 
#         d['Predcited'].append(answer)
#         d['Ground Truth'].append(ans)
#         d['Match'].append(flag)
#         total[ans] += 1

# df = pd.DataFrame(d)

# # Step 2: Save the DataFrame to a CSV file
# df.to_csv('output.csv', index=False)

# acc = (sum(count.values()))/(sum(total.values()))
# print(acc)



import pandas as pd
import re
from collections import defaultdict

gd = pd.read_csv('/scratch/sai/train.csv')
# gd['filename']=gd['filename'].apply(lambda x: x.split('/')[-1])
d = defaultdict(list)



import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the class labels and their corresponding indices
target_labels = {
    "straight": 0,
    "slow down": 1,
    "left turn": 2,
    "left lane change": 3,
    "right turn": 4,
    "right lane change": 5,
    "u turn": 6
}

# Initialize loss comparison dictionaries
predicted = {}
loss_dict = {}

# Load model and tokenizer
model_name = "meta-llama/Meta-Llama-3.1-8B"  # Replace with the correct model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to('cpu')  # Move model to GPU if available

# Set pad_token to eos_token if it's missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

for filename, reponse in video_responses.items():
    
    new_rep = []
    for res in reponse:
        if ".txt" in res:
            continue
        else:  
            res = re.sub(r'\d+: #C', '', res)
            new_rep.append(res)



    # Generate the input prompt by appending each target label as #Caption
    input_prompt = '#Caption - ' + ''.join(new_rep)

    # Iterate through each target label and calculate the loss for each one
    for target_label_text, target_label_idx in target_labels.items():
        # Construct the full prompt for each label
        full_prompt =  f''' Based on the narration of the driving video marked as #Caption, classify the maneuver into one of these classes: "straight", "slow down", "left turn", "left lane change", "right turn", "right lane change", "u turn". Generate only the  maneuver itself at the end of the sentence without repeating the input narration.''' + input_prompt

        # Tokenize the input text and the target label
        inputs = tokenizer(full_prompt, return_tensors="pt").to('cpu')
        target_labels_tokenized = tokenizer(target_label_text, return_tensors="pt").to('cpu')

        # Pad the target to match input length
        target_labels_tokenized['input_ids'] = torch.nn.functional.pad(
            target_labels_tokenized['input_ids'], 
            (0, inputs['input_ids'].shape[1] - target_labels_tokenized['input_ids'].shape[1]), 
            value=tokenizer.pad_token_id
        )

        # Replace padding tokens in labels with -100 to ignore them in the loss calculation
        target_labels_tokenized['input_ids'][target_labels_tokenized['input_ids'] == tokenizer.pad_token_id] = -100

        # Forward pass through the model with input and target labels to compute loss
        outputs = model(input_ids=inputs["input_ids"], labels=target_labels_tokenized["input_ids"])

        import pdb;pdb.set_trace()    
        # Calculate the loss for this target label
        loss = outputs.loss

        # Save the loss value for each target label
        loss_dict[target_label_text] = loss.item()

    # Find the maneuver with the least loss
    answer = min(loss_dict, key=loss_dict.get)


    if any(gd['name']==filename):
        flag = False
        ans = classes[gd[gd['name']==filename]['class'].to_list()[0]]
        if answer == ans:
            count[ans] += 1
            flag = True
        else:
            print(answer , ans)
        total[ans] += 1
        d['filename'].append(filename) 
        d['Predcited'].append(answer)
        d['Ground Truth'].append(ans)
        d['Match'].append(flag)
        total[ans] += 1

    # print(f"\nPredicted Maneuver with the least loss: {answer} , Target - Lel ")


df = pd.DataFrame(d)

# Step 2: Save the DataFrame to a CSV file
df.to_csv('output.csv', index=False)

acc = (sum(count.values()))/(sum(total.values()))
print(acc)

