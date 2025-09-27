import tkinter as tk
from tkinter import filedialog, messagebox, Frame, Label, Button, Canvas, Scrollbar, Toplevel
from tkinter import ttk
from PIL import Image, ImageTk
import os
import face_recognition
import numpy as np
import threading

class FaceSelectionDialog:
    def __init__(self, parent, face_images, face_locations):
        self.parent = parent
        self.face_images = face_images
        self.face_locations = face_locations
        self.selected_face_index = None
        
        self.dialog = Toplevel(parent)
        self.dialog.title("Select Reference Face")
        self.dialog.geometry("600x400")
        self.dialog.configure(bg="#f0f0f0")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.dialog.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.dialog.winfo_height()) // 2
        self.dialog.geometry(f"+{x}+{y}")
        
        self.create_widgets()
        
    def create_widgets(self):
        title = Label(self.dialog, text="Multiple faces detected. Please select which face to use:", 
                     font=("Arial", 14, "bold"), bg="#f0f0f0")
        title.pack(pady=10)
        
        instruction = Label(self.dialog, text="Click on the face you want to use for similarity matching", 
                           font=("Arial", 10), bg="#f0f0f0")
        instruction.pack(pady=5)
        
        # Canvas for scrollable face previews
        canvas_frame = Frame(self.dialog, bg="#f0f0f0")
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        canvas = Canvas(canvas_frame, bg="#f0f0f0")
        scrollbar = Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas, bg="#f0f0f0")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Display face previews
        self.face_photos = []
        self.face_buttons = []
        
        cols = 3
        for idx, face_img in enumerate(self.face_images):
            face_frame = Frame(scrollable_frame, bg="#ffffff", relief="raised", borderwidth=1, padx=5, pady=5)
            face_frame.grid(row=idx // cols, column=idx % cols, padx=10, pady=10, sticky="nsew")
            face_frame.columnconfigure(0, weight=1)
            
            # Create thumbnail
            face_img.thumbnail((150, 150), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(face_img)
            self.face_photos.append(photo)
            
            # Clickable face button
            face_btn = Button(face_frame, image=photo, command=lambda i=idx: self.select_face(i),
                             relief="flat", bg="#ffffff", cursor="hand2")
            face_btn.pack(pady=5)
            self.face_buttons.append(face_btn)
            
            # Face number label
            Label(face_frame, text=f"Face {idx + 1}", font=("Arial", 10, "bold"), 
                 bg="#ffffff").pack(pady=2)
        
        # OK button (initially disabled)
        self.ok_btn = Button(self.dialog, text="Use Selected Face", command=self.ok_clicked,
                           font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", 
                           state="disabled", padx=20, pady=10)
        self.ok_btn.pack(pady=10)
        
    def select_face(self, index):
        self.selected_face_index = index
        self.ok_btn.config(state="normal", bg="#45a049")
        
        # Highlight selected face
        for i, btn in enumerate(self.face_buttons):
            if i == index:
                btn.config(relief="solid", borderwidth=2, bg="#e0e0e0")
            else:
                btn.config(relief="flat", borderwidth=0, bg="#ffffff")
    
    def ok_clicked(self):
        if self.selected_face_index is not None:
            self.dialog.destroy()
    
    def wait_for_selection(self):
        self.parent.wait_window(self.dialog)
        return self.selected_face_index

class FaceSimilarityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Similarity Finder")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        self.folder_path = ""
        self.folder_face_data = []  # Store face data for each image
        self.tolerance = tk.DoubleVar(value=0.5)
        self.reference_encoding = None
        self.selected_images = set()  # Track selected images
        self.current_matches = []  # Store current match results

        # Header
        header = Label(root, text="Face Similarity Finder", font=("Arial", 18, "bold"), 
                      bg="#f0f0f0", fg="#333")
        header.pack(pady=20)

        # Controls Frame
        controls = Frame(root, bg="#f0f0f0")
        controls.pack(pady=10)

        self.btn_select_folder = Button(controls, text="Select Folder with Images", 
                                       command=self.select_folder, font=("Arial", 12, "bold"),
                                       bg="#4CAF50", fg="white", padx=20, pady=10)
        self.btn_select_folder.grid(row=0, column=0, padx=20)

        self.btn_select_query = Button(controls, text="Select Reference Image", 
                                      command=self.select_query_image, font=("Arial", 12, "bold"),
                                      bg="#2196F3", fg="white", padx=20, pady=10)
        self.btn_select_query.grid(row=0, column=1, padx=20)
        self.btn_select_query.config(state="disabled")

        # Tolerance Slider
        Label(controls, text="Tolerance (0.4-0.6):", font=("Arial", 12), 
             bg="#f0f0f0").grid(row=0, column=2, padx=10)
        tolerance_slider = tk.Scale(controls, from_=0.4, to=0.6, variable=self.tolerance, 
                                   orient="horizontal", length=200, resolution=0.01,
                                   bg="#f0f0f0", font=("Arial", 10))
        tolerance_slider.grid(row=0, column=3, padx=10)

        # Selection Controls
        self.btn_use_selected = Button(controls, text="Use Selected Images as New Reference", 
                                      command=self.use_selected_as_reference, font=("Arial", 10, "bold"),
                                      bg="#FF9800", fg="white", padx=15, pady=5, state="disabled")
        self.btn_use_selected.grid(row=0, column=4, padx=20)

        # Progress Bar
        self.progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="indeterminate")
        self.progress.pack(pady=10)
        self.progress.stop()

        # Status Label
        self.status_label = Label(root, text="", font=("Arial", 11), bg="#f0f0f0", fg="#666")
        self.status_label.pack(pady=5)

        # Matches Frame
        self.matches_frame = Frame(root, bg="#ffffff", relief="sunken", borderwidth=2)
        self.matches_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.match_photos = []
        self.image_frames = {}  # Store reference to image frames for selection highlighting

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing photos")
        if folder:
            self.folder_path = folder
            self.selected_images.clear()
            self.status_label.config(text="Loading and processing images...")
            self.progress.start()
            threading.Thread(target=self.load_folder_faces, daemon=True).start()

    def load_folder_faces(self):
        self.folder_face_data.clear()

        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(valid_ext)]
        
        processed = 0
        total = len(files)

        for file in files:
            path = os.path.join(self.folder_path, file)
            try:
                # Load image and detect faces
                img = face_recognition.load_image_file(path)
                face_locations = face_recognition.face_locations(img)
                
                # For each face location, crop the face and encode it
                face_encodings = []
                for location in face_locations:
                    # Crop face region for encoding
                    top, right, bottom, left = location
                    face_img = img[top:bottom, left:right]
                    encoding = face_recognition.face_encodings(face_img)
                    if encoding:
                        face_encodings.append(encoding[0])
                    else:
                        # If encoding fails on cropped face, try with original image
                        encoding = face_recognition.face_encodings(img, [location])
                        if encoding:
                            face_encodings.append(encoding[0])
                
                if face_encodings:
                    self.folder_face_data.append({
                        'path': path,
                        'encodings': face_encodings,
                        'locations': face_locations
                    })
                
                processed += 1
                # Update progress periodically
                if processed % 5 == 0:
                    self.root.after(0, lambda p=processed, t=total: 
                                  self.status_label.config(text=f"Processed {p}/{t} images..."))
                    
            except Exception as e:
                print(f"Error processing {file}: {e}")

        self.root.after(0, self.finish_loading)

    def finish_loading(self):
        self.progress.stop()
        total_faces = sum(len(data['encodings']) for data in self.folder_face_data)
        messagebox.showinfo("Folder Loaded", 
                          f"Processed {len(self.folder_face_data)} images with {total_faces} total faces detected.")
        self.btn_select_query.config(state="normal")
        self.status_label.config(text=f"Ready - {len(self.folder_face_data)} images loaded with {total_faces} faces")

    def select_query_image(self):
        path = filedialog.askopenfilename(title="Select reference image",
                                        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return

        try:
            # Load and detect faces in reference image
            img = face_recognition.load_image_file(path)
            face_locations = face_recognition.face_locations(img)
            
            # Crop and encode each face individually
            face_encodings = []
            for location in face_locations:
                top, right, bottom, left = location
                face_img = img[top:bottom, left:right]
                encoding = face_recognition.face_encodings(face_img)
                if encoding:
                    face_encodings.append(encoding[0])
                else:
                    # Fallback to original image encoding
                    encoding = face_recognition.face_encodings(img, [location])
                    if encoding:
                        face_encodings.append(encoding[0])
            
            if not face_encodings:
                messagebox.showwarning("No face detected", "No faces detected in reference image.")
                return
            
            # If only one face, use it directly
            if len(face_encodings) == 1:
                self.reference_encoding = face_encodings[0]
                self.start_matching_process()
            else:
                # Multiple faces - let user choose
                self.show_face_selection_dialog(path, face_locations, face_encodings)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error processing reference image: {e}")

    def show_face_selection_dialog(self, image_path, face_locations, face_encodings):
        """Show dialog for user to select which face to use from reference image"""
        try:
            # Load the image with PIL for cropping
            pil_image = Image.open(image_path)
            
            # Extract face regions
            face_images = []
            for location in face_locations:
                top, right, bottom, left = location
                # Expand crop area slightly for better preview
                margin = 20
                expanded_top = max(0, top - margin)
                expanded_bottom = min(pil_image.height, bottom + margin)
                expanded_left = max(0, left - margin)
                expanded_right = min(pil_image.width, right + margin)
                
                face_img = pil_image.crop((expanded_left, expanded_top, expanded_right, expanded_bottom))
                face_images.append(face_img)
            
            # Show selection dialog
            dialog = FaceSelectionDialog(self.root, face_images, face_locations)
            selected_index = dialog.wait_for_selection()
            
            if selected_index is not None:
                self.reference_encoding = face_encodings[selected_index]
                self.start_matching_process()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error showing face selection: {e}")

    def start_matching_process(self):
        """Start the face matching process with the selected reference face"""
        self.selected_images.clear()
        self.btn_use_selected.config(state="disabled")
        self.status_label.config(text="Finding similar faces...")
        self.progress.start()
        threading.Thread(target=self.process_matches, daemon=True).start()

    def process_matches(self):
        """Find similar faces using cropped face regions"""
        matches = self.find_similar_faces(self.reference_encoding, self.tolerance.get())
        self.current_matches = matches
        self.root.after(0, lambda: self.show_matches(matches))

    def find_similar_faces(self, query_encoding, tolerance):
        """Find similar faces by comparing cropped face encodings"""
        matched_paths = []
        
        for image_data in self.folder_face_data:
            path = image_data['path']
            face_encodings = image_data['encodings']
            face_locations = image_data['locations']
            
            # Compare each face in the image with the reference face
            best_distance = float('inf')
            best_face_index = -1
            
            for i, encoding in enumerate(face_encodings):
                distance = np.linalg.norm(encoding - query_encoding)
                if distance < best_distance:
                    best_distance = distance
                    best_face_index = i
            
            # If best match is within tolerance, add to results
            if best_distance <= tolerance and best_face_index != -1:
                matched_paths.append({
                    'path': path,
                    'distance': best_distance,
                    'face_index': best_face_index,
                    'total_faces_in_image': len(face_encodings)
                })
        
        # Sort by similarity (lowest distance first)
        matched_paths.sort(key=lambda x: x['distance'])
        return matched_paths

    def show_matches(self, matched_paths):
        """Display the matching results with FULL original images"""
        self.progress.stop()
        
        # Clear previous matches
        for widget in self.matches_frame.winfo_children():
            widget.destroy()
        self.match_photos.clear()
        self.image_frames.clear()

        if not matched_paths:
            Label(self.matches_frame, text="No matching faces found.", 
                 font=("Arial", 14), bg="#ffffff").pack(pady=20)
            self.status_label.config(text="No matches found")
            return

        self.status_label.config(text=f"Found {len(matched_paths)} matching images - Click images to select")

        # Header
        header = Label(self.matches_frame, text=f"Found {len(matched_paths)} matching images:", 
                      font=("Arial", 16, "bold"), bg="#ffffff")
        header.pack(pady=10)

        # Selection info
        selection_info = Label(self.matches_frame, text="Click images to select/deselect (selected images have blue border)", 
                              font=("Arial", 10), bg="#ffffff", fg="#666")
        selection_info.pack(pady=5)

        # Create scrollable canvas for results
        canvas = Canvas(self.matches_frame, bg="#ffffff")
        scrollbar = Scrollbar(self.matches_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas, bg="#ffffff")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Display matches with FULL original images (no bounding boxes)
        cols = 4
        for idx, match_data in enumerate(matched_paths):
            path = match_data['path']
            distance = match_data['distance']
            face_index = match_data['face_index']
            total_faces = match_data['total_faces_in_image']

            # Create clickable image frame
            img_frame = Frame(scrollable_frame, bg="#f0f0f0", relief="raised", 
                             borderwidth=2, padx=5, pady=5)
            img_frame.grid(row=idx // cols, column=idx % cols, padx=10, pady=10, sticky="nsew")
            
            # Store reference to this frame
            self.image_frames[path] = img_frame

            try:
                # Load and display the FULL original image (no bounding boxes)
                img = Image.open(path)
                
                # Create thumbnail of the full image
                img_thumbnail = img.copy()
                img_thumbnail.thumbnail((200, 200), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(img_thumbnail)
                self.match_photos.append(photo)
                
                # Create clickable image label
                img_label = Label(img_frame, image=photo, bg="#f0f0f0", cursor="hand2")
                img_label.pack(pady=5)
                
                # Bind click event to select/deselect image
                img_label.bind("<Button-1>", lambda e, p=path: self.toggle_image_selection(p))
                img_frame.bind("<Button-1>", lambda e, p=path: self.toggle_image_selection(p))

                similarity = (1 - distance) * 100
                
                # Create info text
                info_text = f"{os.path.basename(path)}\nSimilarity: {similarity:.1f}%"
                if total_faces > 1:
                    info_text += f"\n(Match: Face {face_index + 1} of {total_faces})"
                
                file_label = Label(img_frame, text=info_text, 
                                 font=("Arial", 10), bg="#f0f0f0", wraplength=200, justify="center")
                file_label.pack(pady=5)
                file_label.bind("<Button-1>", lambda e, p=path: self.toggle_image_selection(p))
                
            except Exception as e:
                print(f"Error displaying match {path}: {e}")
                error_label = Label(img_frame, text=f"Error loading image\n{os.path.basename(path)}", 
                                  font=("Arial", 10), bg="#f0f0f0", fg="red")
                error_label.pack(pady=20)

        # Enable selection button if we have matches
        if matched_paths:
            self.btn_use_selected.config(state="normal")

    def toggle_image_selection(self, image_path):
        """Toggle selection state of an image"""
        if image_path in self.selected_images:
            self.selected_images.remove(image_path)
            # Remove highlight
            self.image_frames[image_path].config(relief="raised", borderwidth=2, bg="#f0f0f0")
        else:
            self.selected_images.add(image_path)
            # Add highlight
            self.image_frames[image_path].config(relief="solid", borderwidth=3, bg="#e3f2fd")
        
        # Update status
        self.status_label.config(text=f"Selected {len(self.selected_images)} images - Click 'Use Selected Images' to search for similar faces")

    def use_selected_as_reference(self):
        """Use the selected images to create a new reference encoding"""
        if not self.selected_images:
            messagebox.showwarning("No Selection", "Please select one or more images first.")
            return
        
        if len(self.selected_images) == 1:
            # Single image selected - use its best face
            selected_path = next(iter(self.selected_images))
            self.use_single_image_as_reference(selected_path)
        else:
            # Multiple images selected - combine encodings
            self.use_multiple_images_as_reference()

    def use_single_image_as_reference(self, image_path):
        """Use a single selected image as reference"""
        try:
            # Find the image data
            image_data = next((data for data in self.folder_face_data if data['path'] == image_path), None)
            if not image_data:
                messagebox.showerror("Error", "Could not find selected image data.")
                return
            
            if len(image_data['encodings']) == 1:
                self.reference_encoding = image_data['encodings'][0]
            else:
                # Multiple faces - let user choose
                img = face_recognition.load_image_file(image_path)
                face_locations = image_data['locations']
                self.show_face_selection_dialog(image_path, face_locations, image_data['encodings'])
                return
            
            self.start_matching_process()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error using selected image as reference: {e}")

    def use_multiple_images_as_reference(self):
        """Combine encodings from multiple selected images to create an average reference"""
        try:
            all_encodings = []
            for image_path in self.selected_images:
                image_data = next((data for data in self.folder_face_data if data['path'] == image_path), None)
                if image_data and image_data['encodings']:
                    # Use the first face encoding from each selected image
                    all_encodings.append(image_data['encodings'][0])
            
            if not all_encodings:
                messagebox.showerror("Error", "No valid face encodings found in selected images.")
                return
            
            # Create average encoding
            self.reference_encoding = np.mean(all_encodings, axis=0)
            self.start_matching_process()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error combining selected images: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceSimilarityApp(root)
    root.mainloop()