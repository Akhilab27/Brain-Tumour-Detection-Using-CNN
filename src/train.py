from model import build_model
from dataset_loader import train_generator, val_generator

model = build_model()
model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20
)
model.save("brain_tumor_cnn_model.h5") # Here we are saving the model for prediction and testing 

loss, accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

