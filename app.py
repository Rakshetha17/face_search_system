import tkinter as tk
from tkinter import filedialog, messagebox, Frame, Label, Button, Canvas, Scrollbar, Toplevel
from tkinter import ttk
from PIL import Image, ImageTk
import os
import shutil
import face_recognition
import numpy as np
import threading

class MultiFaceSelectionDialog:
    def __init__(self, parent, image_path, face_locations, face_encodings):
        self.parent = parent
        self.image_path = image_path
        self.face_locations = face_locations
        self.face_encodings = face_encodings
        self.selected_face_indices = []  # Store selected face indices
        
        self.dialog = Toplevel(parent)
        self.dialog.title("Select Faces for Matching")
        self.dialog.geometry("750x650")
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
        main_frame = Frame(self.dialog, bg="#f0f0f0")
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Title
        title = Label(main_frame, text="Multiple Faces Detected - Select Faces for Matching", 
                     font=("Arial", 14, "bold"), bg="#f0f0f0")
        title.pack(pady=10)
        
        # Instructions
        instruction_text = (
            "Select which faces to use for similarity matching:\n"
            "• Select one face → match images containing that specific face\n"
            "• Select multiple faces → match images containing ANY of the selected faces"
        )
        instruction = Label(main_frame, text=instruction_text,
                           font=("Arial", 10), bg="#f0f0f0", justify="left")
        instruction.pack(pady=5)
        
        # Original image preview
        orig_frame = Frame(main_frame, bg="#ffffff", relief="sunken", borderwidth=1)
        orig_frame.pack(pady=10, fill="x", padx=20)
        
        try:
            orig_img = Image.open(self.image_path)
            orig_img.thumbnail((300, 200), Image.Resampling.LANCZOS)
            self.orig_photo = ImageTk.PhotoImage(orig_img)
            orig_label = Label(orig_frame, image=self.orig_photo, bg="#ffffff")
            orig_label.pack(pady=5)
            
            filename_label = Label(orig_frame, text=f"Reference: {os.path.basename(self.image_path)}", 
                                  font=("Arial", 9), bg="#ffffff")
            filename_label.pack(pady=2)
        except Exception as e:
            Label(orig_frame, text="Could not load original image", bg="#ffffff").pack(pady=5)
        
        # Faces selection area
        faces_label = Label(main_frame, text="Detected Faces (click to select/deselect):", 
                           font=("Arial", 11, "bold"), bg="#f0f0f0")
        faces_label.pack(pady=(15, 5))
        
        # Canvas for scrollable face previews
        canvas_frame = Frame(main_frame, bg="#f0f0f0", height=250)
        canvas_frame.pack(fill="both", expand=True, pady=5)
        canvas_frame.pack_propagate(False)
        
        canvas = Canvas(canvas_frame, bg="#f0f0f0")
        scrollbar = Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas, bg="#f0f0f0")
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Display face previews
        self.face_photos = []
        self.face_buttons = []
        
        try:
            pil_image = Image.open(self.image_path)
            cols = 4
            
            for idx, location in enumerate(self.face_locations):
                top, right, bottom, left = location
                
                # Expand crop area slightly for better preview
                margin = 15
                expanded_top = max(0, top - margin)
                expanded_bottom = min(pil_image.height, bottom + margin)
                expanded_left = max(0, left - margin)
                expanded_right = min(pil_image.width, right + margin)
                
                face_img = pil_image.crop((expanded_left, expanded_top, expanded_right, expanded_bottom))
                
                face_frame = Frame(scrollable_frame, bg="#ffffff", relief="raised", 
                                 borderwidth=1, padx=5, pady=5)
                face_frame.grid(row=idx // cols, column=idx % cols, padx=8, pady=8, sticky="nsew")
                
                # Create thumbnail
                face_img.thumbnail((100, 100), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(face_img)
                self.face_photos.append(photo)
                
                # Clickable face button
                face_btn = Button(face_frame, image=photo, 
                                command=lambda i=idx: self.toggle_face_selection(i),
                                relief="flat", bg="#ffffff", cursor="hand2")
                face_btn.pack(pady=3)
                self.face_buttons.append(face_btn)
                
                # Face number and selection indicator
                face_info_frame = Frame(face_frame, bg="#ffffff")
                face_info_frame.pack(fill="x", pady=2)
                
                face_label = Label(face_info_frame, text=f"Face {idx + 1}", 
                                 font=("Arial", 9, "bold"), bg="#ffffff")
                face_label.pack(side="left")
                
                # Selection indicator (initially hidden)
                selection_indicator = Label(face_info_frame, text="✓", 
                                          font=("Arial", 12, "bold"), bg="#ffffff", fg="green")
                selection_indicator.pack(side="right")
                selection_indicator.pack_forget()  # Hide initially
                
                # Store indicator reference
                face_btn.selection_indicator = selection_indicator
                
        except Exception as e:
            error_label = Label(scrollable_frame, text=f"Error loading faces: {str(e)}", 
                              font=("Arial", 10), bg="#f0f0f0")
            error_label.pack(pady=20)
        
        # Selection control buttons
        control_frame = Frame(main_frame, bg="#f0f0f0")
        control_frame.pack(pady=10)
        
        select_all_btn = Button(control_frame, text="Select All Faces", 
                              command=self.select_all_faces,
                              font=("Arial", 10), bg="#2196F3", fg="white", padx=10, pady=3)
        select_all_btn.pack(side="left", padx=5)
        
        clear_btn = Button(control_frame, text="Clear Selection", 
                         command=self.clear_selection,
                         font=("Arial", 10), bg="#FF9800", fg="white", padx=10, pady=3)
        clear_btn.pack(side="left", padx=5)
        
        # Action buttons
        button_frame = Frame(main_frame, bg="#f0f0f0")
        button_frame.pack(pady=15)
        
        # Use Selected Faces button (initially disabled)
        self.use_btn = Button(button_frame, text="Use Selected Faces for Matching", 
                            command=self.use_selected_faces,
                            font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", 
                            state="disabled", padx=20, pady=6)
        self.use_btn.pack(side="left", padx=10)
        
        cancel_btn = Button(button_frame, text="Cancel", 
                          command=self.cancel,
                          font=("Arial", 10), bg="#f44336", fg="white", padx=15, pady=4)
        cancel_btn.pack(side="left", padx=10)
        
    def toggle_face_selection(self, index):
        """Toggle selection for a specific face"""
        if index in self.selected_face_indices:
            # Deselect face
            self.selected_face_indices.remove(index)
            self.face_buttons[index].config(relief="flat", bg="#ffffff")
            self.face_buttons[index].selection_indicator.pack_forget()
        else:
            # Select face
            self.selected_face_indices.append(index)
            self.face_buttons[index].config(relief="solid", bg="#e3f2fd", borderwidth=3)
            self.face_buttons[index].selection_indicator.pack(side="right")
        
        self.update_use_button()
    
    def select_all_faces(self):
        """Select all faces"""
        self.selected_face_indices = list(range(len(self.face_locations)))
        for i, btn in enumerate(self.face_buttons):
            btn.config(relief="solid", bg="#e3f2fd", borderwidth=3)
            btn.selection_indicator.pack(side="right")
        self.update_use_button()
    
    def clear_selection(self):
        """Clear all selections"""
        self.selected_face_indices.clear()
        for btn in self.face_buttons:
            btn.config(relief="flat", bg="#ffffff")
            btn.selection_indicator.pack_forget()
        self.update_use_button()
    
    def update_use_button(self):
        """Update the use button text and state based on selection"""
        selected_count = len(self.selected_face_indices)
        
        if selected_count > 0:
            self.use_btn.config(state="normal", bg="#45a049")
            if selected_count == 1:
                face_num = self.selected_face_indices[0] + 1
                self.use_btn.config(text=f"Use Face {face_num} for Matching")
            else:
                self.use_btn.config(text=f"Use {selected_count} Faces for Matching")
        else:
            self.use_btn.config(state="disabled", bg="#4CAF50")
            self.use_btn.config(text="Use Selected Faces for Matching")
    
    def use_selected_faces(self):
        """Confirm selection and close dialog"""
        if self.selected_face_indices:
            self.dialog.destroy()
        else:
            messagebox.showwarning("No Selection", "Please select at least one face to continue.")
    
    def cancel(self):
        """Cancel selection"""
        self.selected_face_indices.clear()
        self.dialog.destroy()
    
    def wait_for_selection(self):
        """Wait for user selection and return selected face encodings"""
        self.parent.wait_window(self.dialog)
        
        if self.selected_face_indices:
            # Return the encodings of selected faces
            selected_encodings = [self.face_encodings[i] for i in sorted(self.selected_face_indices)]
            return selected_encodings
        return None

class FaceSimilarityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Face Similarity Finder with Multi-Face Selection")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        self.folder_path = ""
        self.folder_face_data = []
        self.tolerance = tk.DoubleVar(value=0.5)
        self.reference_encodings = []
        self.current_matches = []
        self.search_history = []
        self.selected_images = set()

        # Header
        header = Label(root, text="Advanced Face Similarity Finder with Multi-Face Selection", 
                      font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#333")
        header.pack(pady=20)

        # Controls Frame
        controls = Frame(root, bg="#f0f0f0")
        controls.pack(pady=10)

        # Row 1: Main controls
        self.btn_select_folder = Button(controls, text="📁 Select Folder with Images", 
                                       command=self.select_folder, font=("Arial", 12, "bold"),
                                       bg="#4CAF50", fg="white", padx=20, pady=10)
        self.btn_select_folder.grid(row=0, column=0, padx=15, pady=5)

        self.btn_select_query = Button(controls, text="🔍 Select Reference Image", 
                                      command=self.select_query_image, font=("Arial", 12, "bold"),
                                      bg="#2196F3", fg="white", padx=20, pady=10)
        self.btn_select_query.grid(row=0, column=1, padx=15, pady=5)
        self.btn_select_query.config(state="disabled")

        # Tolerance Slider
        Label(controls, text="Tolerance (0.4-0.6):", font=("Arial", 12), 
             bg="#f0f0f0").grid(row=0, column=2, padx=10, pady=5)
        tolerance_slider = tk.Scale(controls, from_=0.4, to=0.6, variable=self.tolerance, 
                                   orient="horizontal", length=200, resolution=0.01,
                                   bg="#f0f0f0", font=("Arial", 10))
        tolerance_slider.grid(row=0, column=3, padx=10, pady=5)

        # Row 2: Action buttons
        self.btn_back = Button(controls, text="↶ Back to Previous Search", 
                              command=self.navigate_back, font=("Arial", 10),
                              bg="#607D8B", fg="white", padx=15, pady=6, state="disabled")
        self.btn_back.grid(row=1, column=0, padx=15, pady=5)

        self.btn_use_selected = Button(controls, text="⭐ Use Selected as Reference", 
                                      command=self.use_selected_as_reference, font=("Arial", 10, "bold"),
                                      bg="#9C27B0", fg="white", padx=15, pady=6, state="disabled")
        self.btn_use_selected.grid(row=1, column=1, padx=15, pady=5)

        self.btn_copy = Button(controls, text="📋 Copy Selected", 
                              command=self.copy_selected_files, font=("Arial", 10, "bold"),
                              bg="#009688", fg="white", padx=15, pady=6, state="disabled")
        self.btn_copy.grid(row=1, column=2, padx=15, pady=5)

        self.btn_cut = Button(controls, text="✂️ Cut Selected", 
                             command=self.cut_selected_files, font=("Arial", 10, "bold"),
                             bg="#FF5722", fg="white", padx=15, pady=6, state="disabled")
        self.btn_cut.grid(row=1, column=3, padx=15, pady=5)

        # Row 3: Selection management
        self.btn_select_all = Button(controls, text="✓ Select All Results", 
                                    command=self.select_all_results, font=("Arial", 10),
                                    bg="#3F51B5", fg="white", padx=15, pady=5, state="disabled")
        self.btn_select_all.grid(row=2, column=0, padx=15, pady=5)

        self.btn_clear_selection = Button(controls, text="✗ Clear Selection", 
                                         command=self.clear_selection, font=("Arial", 10),
                                         bg="#f44336", fg="white", padx=15, pady=5, state="disabled")
        self.btn_clear_selection.grid(row=2, column=1, padx=15, pady=5)

        # Progress Bar
        self.progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="indeterminate")
        self.progress.pack(pady=10)
        self.progress.stop()

        # Status Label
        self.status_label = Label(root, text="", font=("Arial", 11), bg="#f0f0f0", fg="#666")
        self.status_label.pack(pady=5)

        # Instruction Label
        self.instruction_label = Label(root, text="", font=("Arial", 10, "italic"), 
                                      bg="#f0f0f0", fg="#2196F3")
        self.instruction_label.pack(pady=2)

        # Matches Frame
        self.matches_frame = Frame(root, bg="#ffffff", relief="sunken", borderwidth=2)
        self.matches_frame.pack(fill="both", expand=True, padx=20, pady=15)

        self.match_photos = []
        self.image_frames = {}

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing photos")
        if folder:
            self.folder_path = folder
            self.reference_encodings.clear()
            self.search_history.clear()
            self.selected_images.clear()
            self.update_ui_state()
            self.btn_back.config(state="disabled")
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
                img = face_recognition.load_image_file(path)
                face_locations = face_recognition.face_locations(img)
                
                # Use cropped faces for encoding (internal matching)
                face_encodings = []
                for location in face_locations:
                    top, right, bottom, left = location
                    face_img = img[top:bottom, left:right]
                    encoding = face_recognition.face_encodings(face_img)
                    if encoding:
                        face_encodings.append(encoding[0])
                    else:
                        # Fallback to full image encoding if cropped face fails
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
            img = face_recognition.load_image_file(path)
            face_locations = face_recognition.face_locations(img)
            
            # Use cropped faces for encoding (internal matching)
            face_encodings = []
            for location in face_locations:
                top, right, bottom, left = location
                face_img = img[top:bottom, left:right]
                encoding = face_recognition.face_encodings(face_img)
                if encoding:
                    face_encodings.append(encoding[0])
                else:
                    encoding = face_recognition.face_encodings(img, [location])
                    if encoding:
                        face_encodings.append(encoding[0])
            
            if not face_encodings:
                messagebox.showwarning("No face detected", "No faces detected in reference image.")
                return
            
            # Save current state to history
            if self.reference_encodings:
                self.search_history.append(self.reference_encodings.copy())
                self.btn_back.config(state="normal")
            
            # Show face selection dialog if multiple faces detected
            if len(face_encodings) > 1:
                self.show_multi_face_selection(path, face_locations, face_encodings)
            else:
                # Single face - use it directly
                self.reference_encodings = face_encodings
                self.status_label.config(text="Using single face for matching...")
                self.start_matching_process()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error processing reference image: {e}")

    def show_multi_face_selection(self, image_path, face_locations, face_encodings):
        """Show dialog for selecting which faces to use from reference image"""
        try:
            dialog = MultiFaceSelectionDialog(self.root, image_path, face_locations, face_encodings)
            selected_encodings = dialog.wait_for_selection()
            
            if selected_encodings:
                self.reference_encodings = selected_encodings
                
                if len(selected_encodings) == 1:
                    self.status_label.config(text="Using selected face for matching...")
                else:
                    self.status_label.config(text=f"Using {len(selected_encodings)} selected faces for matching...")
                
                self.start_matching_process()
            else:
                # User cancelled or closed the dialog
                self.status_label.config(text="Face selection cancelled. Please select a new reference image.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error showing face selection: {e}")

    def start_matching_process(self):
        """Start the face matching process with threading"""
        self.selected_images.clear()
        self.update_ui_state()
        self.status_label.config(text="Finding similar faces...")
        self.progress.start()
        threading.Thread(target=self.process_matches, daemon=True).start()

    def process_matches(self):
        """Process matches in a separate thread"""
        matches = self.find_similar_faces(self.reference_encodings, self.tolerance.get())
        self.current_matches = matches
        self.root.after(0, lambda: self.show_matches(matches))

    def find_similar_faces(self, query_encodings, tolerance):
        """Find similar faces - matches images containing ANY of the selected faces"""
        matched_paths = []
        
        for image_data in self.folder_face_data:
            path = image_data['path']
            face_encodings = image_data['encodings']
            
            # Find the best matching face in this image against all reference faces
            best_distance = float('inf')
            best_match_found = False
            
            for face_encoding in face_encodings:
                for ref_encoding in query_encodings:
                    distance = np.linalg.norm(face_encoding - ref_encoding)
                    if distance < best_distance:
                        best_distance = distance
                    if distance <= tolerance:
                        best_match_found = True
                        break  # No need to check other reference faces for this image face
                if best_match_found:
                    break  # No need to check other faces in this image
            
            # If any face matches within tolerance, include the image
            if best_match_found:
                matched_paths.append({
                    'path': path,
                    'distance': best_distance,
                    'total_faces': len(face_encodings),
                    'ref_faces_used': len(query_encodings)
                })
        
        # Sort by similarity (lowest distance first)
        matched_paths.sort(key=lambda x: x['distance'])
        return matched_paths

    def show_matches(self, matched_paths):
        """Display the matching results"""
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
            self.update_ui_state()
            return

        # Update status with reference face info
        ref_count = len(self.reference_encodings)
        ref_text = f" ({ref_count} reference face{'s' if ref_count > 1 else ''})"
        
        self.status_label.config(text=f"Found {len(matched_paths)} matching images{ref_text}")
        self.instruction_label.config(text="💡 Click images to select | Double-click to use as new reference")

        # Create scrollable results area
        canvas = Canvas(self.matches_frame, bg="#ffffff")
        scrollbar = Scrollbar(self.matches_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas, bg="#ffffff")

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Display matches with FULL original images (no bounding boxes)
        cols = 4
        for idx, match_data in enumerate(matched_paths):
            path = match_data['path']
            distance = match_data['distance']
            total_faces = match_data['total_faces']
            ref_faces_used = match_data.get('ref_faces_used', 1)

            # Create image frame
            img_frame = Frame(scrollable_frame, bg="#f0f0f0", relief="raised", 
                             borderwidth=2, padx=5, pady=5)
            img_frame.grid(row=idx // cols, column=idx % cols, padx=10, pady=10)
            self.image_frames[path] = img_frame

            try:
                # Load and display FULL original image
                img = Image.open(path)
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.match_photos.append(photo)

                # Clickable image label
                img_label = Label(img_frame, image=photo, bg="#f0f0f0", cursor="hand2")
                img_label.pack(pady=5)
                
                # Bind click events
                img_label.bind("<Button-1>", lambda e, p=path: self.toggle_image_selection(p))
                img_label.bind("<Double-Button-1>", lambda e, p=path: self.use_image_as_reference(p))
                img_frame.bind("<Button-1>", lambda e, p=path: self.toggle_image_selection(p))

                # Calculate similarity percentage
                similarity = (1 - distance) * 100
                
                # Create info text
                info_text = f"{os.path.basename(path)}\nSimilarity: {similarity:.1f}%"
                if total_faces > 1:
                    info_text += f"\nFaces: {total_faces}"
                if ref_faces_used > 1:
                    info_text += f"\nMatched {ref_faces_used} ref faces"
                
                file_label = Label(img_frame, text=info_text, 
                                 font=("Arial", 10), bg="#f0f0f0", wraplength=200, 
                                 justify="center", cursor="hand2")
                file_label.pack(pady=5)
                file_label.bind("<Button-1>", lambda e, p=path: self.toggle_image_selection(p))
                
            except Exception as e:
                error_label = Label(img_frame, text=f"Error loading image\n{os.path.basename(path)}", 
                                  font=("Arial", 10), bg="#f0f0f0", fg="red")
                error_label.pack(pady=20)

        self.update_ui_state()

    def toggle_image_selection(self, image_path):
        """Toggle selection state of an image"""
        if image_path in self.selected_images:
            self.selected_images.remove(image_path)
            self.image_frames[image_path].config(bg="#f0f0f0")
        else:
            self.selected_images.add(image_path)
            self.image_frames[image_path].config(bg="#e3f2fd")
        
        self.update_ui_state()
        self.status_label.config(text=f"Selected {len(self.selected_images)} images")

    def use_image_as_reference(self, image_path):
        """Use clicked image as new reference (all faces)"""
        image_data = next((data for data in self.folder_face_data if data['path'] == image_path), None)
        if image_data and image_data['encodings']:
            if self.reference_encodings:
                self.search_history.append(self.reference_encodings.copy())
                self.btn_back.config(state="normal")
            
            # Use ALL faces from the clicked image
            self.reference_encodings = image_data['encodings']
            face_count = len(image_data['encodings'])
            self.status_label.config(text=f"Using {face_count} faces from clicked image for new search...")
            self.start_matching_process()

    def copy_selected_files(self):
        """Copy selected files to destination folder"""
        if not self.selected_images:
            messagebox.showwarning("No Selection", "Please select one or more images first.")
            return
        
        destination = filedialog.askdirectory(title="Select destination folder for copying")
        if not destination:
            return
        
        confirm = messagebox.askyesno("Confirm Copy", f"Copy {len(self.selected_images)} files to selected folder?")
        if not confirm:
            return
        
        success_count = 0
        for image_path in self.selected_images:
            try:
                filename = os.path.basename(image_path)
                dest_path = os.path.join(destination, filename)
                shutil.copy2(image_path, dest_path)
                success_count += 1
            except Exception as e:
                print(f"Error copying {image_path}: {e}")
        
        messagebox.showinfo("Success", f"Successfully copied {success_count} files!")
        self.status_label.config(text=f"Copied {success_count} files to destination")

    def cut_selected_files(self):
        """Remove selected files from results (simulate cut)"""
        if not self.selected_images:
            messagebox.showwarning("No Selection", "Please select one or more images first.")
            return
        
        confirm = messagebox.askyesno("Confirm Remove", 
                                    f"Remove {len(self.selected_images)} selected files from results?")
        if not confirm:
            return
        
        # Remove selected images from current matches
        self.current_matches = [match for match in self.current_matches 
                              if match['path'] not in self.selected_images]
        
        # Clear selection and refresh display
        self.selected_images.clear()
        self.show_matches(self.current_matches)
        self.status_label.config(text="Selected files removed from results")

    def select_all_results(self):
        """Select all displayed images"""
        if self.current_matches:
            self.selected_images.clear()
            for match in self.current_matches:
                self.selected_images.add(match['path'])
            for path in self.selected_images:
                if path in self.image_frames:
                    self.image_frames[path].config(bg="#e3f2fd")
            self.update_ui_state()

    def clear_selection(self):
        """Clear all selections"""
        self.selected_images.clear()
        for path in self.image_frames:
            self.image_frames[path].config(bg="#f0f0f0")
        self.update_ui_state()

    def use_selected_as_reference(self):
        """Use all faces from selected images as new reference"""
        if not self.selected_images:
            messagebox.showwarning("No Selection", "Please select one or more images first.")
            return
        
        # Collect all face encodings from selected images
        all_encodings = []
        for image_path in self.selected_images:
            image_data = next((data for data in self.folder_face_data if data['path'] == image_path), None)
            if image_data:
                all_encodings.extend(image_data['encodings'])
        
        if all_encodings:
            if self.reference_encodings:
                self.search_history.append(self.reference_encodings.copy())
                self.btn_back.config(state="normal")
            
            self.reference_encodings = all_encodings
            self.status_label.config(text=f"Using {len(all_encodings)} faces from selection for new search...")
            self.start_matching_process()

    def navigate_back(self):
        """Navigate back to previous search"""
        if self.search_history:
            self.reference_encodings = self.search_history.pop()
            if not self.search_history:
                self.btn_back.config(state="disabled")
            self.status_label.config(text="Returning to previous search...")
            self.start_matching_process()

    def update_ui_state(self):
        """Update the state of UI elements based on current selection"""
        has_selection = len(self.selected_images) > 0
        has_matches = len(self.current_matches) > 0
        
        self.btn_use_selected.config(state="normal" if has_selection else "disabled")
        self.btn_copy.config(state="normal" if has_selection else "disabled")
        self.btn_cut.config(state="normal" if has_selection else "disabled")
        self.btn_select_all.config(state="normal" if has_matches else "disabled")
        self.btn_clear_selection.config(state="normal" if has_selection else "disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceSimilarityApp(root)
    root.mainloop()