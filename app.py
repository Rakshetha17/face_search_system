import tkinter as tk
from tkinter import filedialog, messagebox, Frame, Label, Button, Canvas, Scrollbar, Toplevel
from tkinter import ttk
from PIL import Image, ImageTk
import os
import shutil
import face_recognition
import numpy as np
import threading

class ImageViewerDialog:
    def __init__(self, parent, image_path, face_data=None):
        self.parent = parent
        self.image_path = image_path
        self.face_data = face_data
        self.selected_faces = set()
        
        self.dialog = Toplevel(parent)
        self.dialog.title(f"Image Viewer - {os.path.basename(image_path)}")
        self.dialog.geometry("900x700")
        self.dialog.configure(bg="#f0f0f0")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.dialog.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.dialog.winfo_height()) // 2
        self.dialog.geometry(f"+{x}+{y}")
        
        # Bind Escape key to close
        self.dialog.bind("<Escape>", lambda e: self.dialog.destroy())
        
        self.create_widgets()
        
    def create_widgets(self):
        main_frame = Frame(self.dialog, bg="#f0f0f0")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header with filename
        filename_label = Label(main_frame, text=os.path.basename(self.image_path), 
                              font=("Arial", 14, "bold"), bg="#f0f0f0")
        filename_label.pack(pady=5)
        
        # Image display area
        img_frame = Frame(main_frame, bg="#ffffff", relief="sunken", borderwidth=1)
        img_frame.pack(fill="both", expand=True, pady=10)
        
        # Canvas with scrollbars for large images
        canvas_frame = Frame(img_frame, bg="#ffffff")
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        h_scrollbar = Scrollbar(canvas_frame, orient="horizontal")
        v_scrollbar = Scrollbar(canvas_frame, orient="vertical")
        
        self.canvas = Canvas(canvas_frame, bg="#ffffff",
                           xscrollcommand=h_scrollbar.set,
                           yscrollcommand=v_scrollbar.set)
        
        h_scrollbar.config(command=self.canvas.xview)
        v_scrollbar.config(command=self.canvas.yview)
        
        h_scrollbar.pack(side="bottom", fill="x")
        v_scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Load and display image
        self.load_image()
        
        # Face selection area (only if multiple faces exist)
        if self.face_data and len(self.face_data['encodings']) > 1:
            self.create_face_selection_area(main_frame)
        
        # Button frame
        button_frame = Frame(main_frame, bg="#f0f0f0")
        button_frame.pack(pady=10)
        
        if self.face_data and len(self.face_data['encodings']) > 1:
            self.use_faces_btn = Button(button_frame, text="Use Selected Faces for New Search", 
                                      command=self.use_selected_faces,
                                      font=("Arial", 11, "bold"), bg="#4CAF50", fg="white", 
                                      padx=15, pady=5)
            self.use_faces_btn.pack(side="left", padx=5)
        
        close_btn = Button(button_frame, text="Close (Esc)", command=self.dialog.destroy,
                         font=("Arial", 10), bg="#607D8B", fg="white", padx=15, pady=5)
        close_btn.pack(side="left", padx=5)
        
    def load_image(self):
        try:
            self.image = Image.open(self.image_path)
            
            # Calculate display size
            display_width = min(self.image.width, 800)
            display_height = min(self.image.height, 500)
            
            if self.image.width > display_width or self.image.height > display_height:
                display_image = self.image.resize((display_width, display_height), Image.Resampling.LANCZOS)
            else:
                display_image = self.image
            
            self.tk_image = ImageTk.PhotoImage(display_image)
            
            # Create canvas image
            self.canvas_image = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
            
            # Update scroll region
            self.canvas.config(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            error_label = Label(self.canvas, text=f"Error loading image:\n{str(e)}", 
                              font=("Arial", 12), bg="#ffffff")
            self.canvas.create_window(400, 250, window=error_label)
    
    def create_face_selection_area(self, parent):
        faces_frame = Frame(parent, bg="#f0f0f0", relief="groove", borderwidth=1)
        faces_frame.pack(fill="x", pady=10, padx=20)
        
        instruction = Label(faces_frame, 
                           text="Multiple faces detected. Select faces to use for new similarity search:",
                           font=("Arial", 10, "bold"), bg="#f0f0f0")
        instruction.pack(pady=5)
        
        # Faces container
        faces_container = Frame(faces_frame, bg="#f0f0f0")
        faces_container.pack(fill="x", pady=5, padx=10)
        
        self.face_buttons = []
        self.face_photos = []
        
        try:
            pil_image = Image.open(self.image_path)
            cols = min(4, len(self.face_data['encodings']))
            
            for idx, location in enumerate(self.face_data['locations']):
                if idx >= 8:  # Limit to 8 faces for UI
                    break
                    
                top, right, bottom, left = location
                margin = 10
                expanded_top = max(0, top - margin)
                expanded_bottom = min(pil_image.height, bottom + margin)
                expanded_left = max(0, left - margin)
                expanded_right = min(pil_image.width, right + margin)
                
                face_img = pil_image.crop((expanded_left, expanded_top, expanded_right, expanded_bottom))
                
                face_frame = Frame(faces_container, bg="#ffffff", relief="raised", borderwidth=1)
                face_frame.grid(row=idx // cols, column=idx % cols, padx=5, pady=5, sticky="w")
                
                face_img.thumbnail((80, 80), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(face_img)
                self.face_photos.append(photo)
                
                face_btn = Button(face_frame, image=photo, 
                                command=lambda i=idx: self.toggle_face_selection(i),
                                relief="flat", bg="#ffffff", cursor="hand2")
                face_btn.pack(pady=2)
                self.face_buttons.append(face_btn)
                
                Label(face_frame, text=f"Face {idx+1}", font=("Arial", 8), bg="#ffffff").pack()
                
        except Exception as e:
            Label(faces_container, text=f"Error loading faces: {str(e)}", bg="#f0f0f0").pack()
    
    def toggle_face_selection(self, index):
        if index in self.selected_faces:
            self.selected_faces.remove(index)
            self.face_buttons[index].config(relief="flat", bg="#ffffff")
        else:
            self.selected_faces.add(index)
            self.face_buttons[index].config(relief="solid", bg="#e3f2fd", borderwidth=2)
    
    def use_selected_faces(self):
        if self.selected_faces:
            self.dialog.destroy()
        else:
            messagebox.showwarning("No Selection", "Please select at least one face.")

class FaceSimilarityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Face Similarity Finder")
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
        header = Label(root, text="Enhanced Face Similarity Finder", 
                      font=("Arial", 18, "bold"), bg="#f0f0f0", fg="#333")
        header.pack(pady=20)

        # Controls Frame
        controls = Frame(root, bg="#f0f0f0")
        controls.pack(pady=10)

        # Row 1: Main controls
        self.btn_select_folder = Button(controls, text="Select Folder with Images", 
                                       command=self.select_folder, font=("Arial", 12, "bold"),
                                       bg="#4CAF50", fg="white", padx=20, pady=10)
        self.btn_select_folder.grid(row=0, column=0, padx=15, pady=5)

        self.btn_select_query = Button(controls, text="Select Reference Image", 
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
        self.btn_back = Button(controls, text="← Back to Previous Search", 
                              command=self.navigate_back, font=("Arial", 10),
                              bg="#607D8B", fg="white", padx=15, pady=6, state="disabled")
        self.btn_back.grid(row=1, column=0, padx=15, pady=5)

        self.btn_use_selected = Button(controls, text="Use Selected Images as New Reference", 
                                      command=self.use_selected_as_reference, font=("Arial", 10, "bold"),
                                      bg="#9C27B0", fg="white", padx=15, pady=6, state="disabled")
        self.btn_use_selected.grid(row=1, column=1, padx=15, pady=5)

        # File management buttons
        self.btn_copy = Button(controls, text="📋 Copy Selected", 
                              command=self.copy_selected_files, font=("Arial", 10, "bold"),
                              bg="#009688", fg="white", padx=15, pady=6, state="disabled")
        self.btn_copy.grid(row=1, column=2, padx=15, pady=5)

        self.btn_cut = Button(controls, text="✂️ Cut Selected", 
                             command=self.cut_selected_files, font=("Arial", 10, "bold"),
                             bg="#FF5722", fg="white", padx=15, pady=6, state="disabled")
        self.btn_cut.grid(row=1, column=3, padx=15, pady=5)

        # Row 3: Selection management
        self.btn_select_all = Button(controls, text="Select All Results", 
                                    command=self.select_all_results, font=("Arial", 10),
                                    bg="#3F51B5", fg="white", padx=15, pady=5, state="disabled")
        self.btn_select_all.grid(row=2, column=0, padx=15, pady=5)

        self.btn_clear_selection = Button(controls, text="Clear Selection", 
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
                
                # Use cropped faces for encoding
                face_encodings = []
                for location in face_locations:
                    top, right, bottom, left = location
                    face_img = img[top:bottom, left:right]
                    encoding = face_recognition.face_encodings(face_img)
                    if encoding:
                        face_encodings.append(encoding[0])
                    else:
                        # Fallback to full image encoding
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
            
            # Use cropped faces for encoding
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
            
            if self.reference_encodings:
                self.search_history.append(self.reference_encodings.copy())
                self.btn_back.config(state="normal")
            
            if len(face_encodings) == 1:
                self.reference_encodings = face_encodings
                self.start_matching_process()
            else:
                # For multiple faces, use all of them
                self.reference_encodings = face_encodings
                self.start_matching_process()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error processing reference image: {e}")

    def start_matching_process(self):
        self.selected_images.clear()
        self.update_ui_state()
        self.status_label.config(text="Finding similar faces...")
        self.progress.start()
        threading.Thread(target=self.process_matches, daemon=True).start()

    def process_matches(self):
        matches = self.find_similar_faces(self.reference_encodings, self.tolerance.get())
        self.current_matches = matches
        self.root.after(0, lambda: self.show_matches(matches))

    def find_similar_faces(self, query_encodings, tolerance):
        matched_paths = []
        
        for image_data in self.folder_face_data:
            path = image_data['path']
            face_encodings = image_data['encodings']
            
            # Find best match among all faces in the image
            best_distance = float('inf')
            for face_encoding in face_encodings:
                for ref_encoding in query_encodings:
                    distance = np.linalg.norm(face_encoding - ref_encoding)
                    if distance < best_distance:
                        best_distance = distance
            
            if best_distance <= tolerance:
                matched_paths.append({
                    'path': path,
                    'distance': best_distance,
                    'total_faces': len(face_encodings)
                })
        
        # Sort by similarity (lowest distance first)
        matched_paths.sort(key=lambda x: x['distance'])
        return matched_paths

    def show_matches(self, matched_paths):
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

        self.status_label.config(text=f"Found {len(matched_paths)} matching images")
        self.instruction_label.config(text="💡 Click images to select | Double-click to view and select faces for new search")

        # Create scrollable results area
        canvas = Canvas(self.matches_frame, bg="#ffffff")
        scrollbar = Scrollbar(self.matches_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas, bg="#ffffff")

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Display matches with full images
        cols = 4
        for idx, match_data in enumerate(matched_paths):
            path = match_data['path']
            distance = match_data['distance']
            total_faces = match_data['total_faces']

            # Create image frame
            img_frame = Frame(scrollable_frame, bg="#f0f0f0", relief="raised", 
                             borderwidth=2, padx=5, pady=5)
            img_frame.grid(row=idx // cols, column=idx % cols, padx=10, pady=10)
            self.image_frames[path] = img_frame

            try:
                # Load and display full image
                img = Image.open(path)
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.match_photos.append(photo)

                # Clickable image label
                img_label = Label(img_frame, image=photo, bg="#f0f0f0", cursor="hand2")
                img_label.pack(pady=5)
                
                # Bind single click for selection, double click for viewer/face selection
                img_label.bind("<Button-1>", lambda e, p=path: self.toggle_image_selection(p))
                img_label.bind("<Double-Button-1>", lambda e, p=path: self.open_image_viewer(p))
                img_frame.bind("<Button-1>", lambda e, p=path: self.toggle_image_selection(p))
                img_frame.bind("<Double-Button-1>", lambda e, p=path: self.open_image_viewer(p))

                similarity = (1 - distance) * 100
                info_text = f"{os.path.basename(path)}\nSimilarity: {similarity:.1f}%"
                if total_faces > 1:
                    info_text += f"\nFaces: {total_faces}"
                
                file_label = Label(img_frame, text=info_text, 
                                 font=("Arial", 10), bg="#f0f0f0", wraplength=200, 
                                 justify="center", cursor="hand2")
                file_label.pack(pady=5)
                file_label.bind("<Button-1>", lambda e, p=path: self.toggle_image_selection(p))
                file_label.bind("<Double-Button-1>", lambda e, p=path: self.open_image_viewer(p))
                
            except Exception as e:
                error_label = Label(img_frame, text=f"Error loading image\n{os.path.basename(path)}", 
                                  font=("Arial", 10), bg="#f0f0f0", fg="red")
                error_label.pack(pady=20)

        self.update_ui_state()

    def toggle_image_selection(self, image_path):
        if image_path in self.selected_images:
            self.selected_images.remove(image_path)
            self.image_frames[image_path].config(bg="#f0f0f0")
        else:
            self.selected_images.add(image_path)
            self.image_frames[image_path].config(bg="#e3f2fd")
        
        self.update_ui_state()
        self.status_label.config(text=f"Selected {len(self.selected_images)} images")

    def open_image_viewer(self, image_path):
        """Open image viewer with face selection for multi-face images"""
        image_data = next((data for data in self.folder_face_data if data['path'] == image_path), None)
        dialog = ImageViewerDialog(self.root, image_path, image_data)
        
        # Wait for dialog to close
        self.root.wait_window(dialog.dialog)
        
        if dialog.selected_faces and image_data:
            # Use selected faces for new search
            if self.reference_encodings:
                self.search_history.append(self.reference_encodings.copy())
                self.btn_back.config(state="normal")
            
            selected_encodings = [image_data['encodings'][i] for i in dialog.selected_faces]
            self.reference_encodings = selected_encodings
            self.status_label.config(text=f"Using {len(selected_encodings)} selected faces for new search...")
            self.start_matching_process()

    def copy_selected_files(self):
        """Copy selected files to destination folder"""
        if not self.selected_images:
            messagebox.showwarning("No Selection", "Please select one or more images first.")
            return
        
        destination = filedialog.askdirectory(title="Select destination folder for copying")
        if not destination:
            return
        
        confirm = messagebox.askyesno(
            "Confirm Copy", 
            f"Copy {len(self.selected_images)} files to:\n{destination}?"
        )
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
        
        messagebox.showinfo("Copy Successful", f"Successfully copied {success_count} files!")
        self.status_label.config(text=f"Copied {success_count} files to destination")

    def cut_selected_files(self):
        """Remove selected files from results (simulate cut operation)"""
        if not self.selected_images:
            messagebox.showwarning("No Selection", "Please select one or more images first.")
            return
        
        confirm = messagebox.askyesno(
            "Confirm Remove", 
            f"Remove {len(self.selected_images)} selected files from results?\n\nNote: Files remain in original folder."
        )
        if not confirm:
            return
        
        # Remove selected images from current matches
        self.current_matches = [match for match in self.current_matches 
                              if match['path'] not in self.selected_images]
        
        # Clear selection
        self.selected_images.clear()
        
        # Refresh display
        self.show_matches(self.current_matches)
        self.status_label.config(text=f"Removed {len(self.current_matches)} files from results")

    def select_all_results(self):
        if self.current_matches:
            self.selected_images.clear()
            for match in self.current_matches:
                self.selected_images.add(match['path'])
            for path in self.selected_images:
                if path in self.image_frames:
                    self.image_frames[path].config(bg="#e3f2fd")
            self.update_ui_state()

    def clear_selection(self):
        self.selected_images.clear()
        for path in self.image_frames:
            self.image_frames[path].config(bg="#f0f0f0")
        self.update_ui_state()

    def use_selected_as_reference(self):
        if not self.selected_images:
            messagebox.showwarning("No Selection", "Please select one or more images first.")
            return
        
        # Use all faces from all selected images
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
            self.status_label.config(text=f"Using {len(all_encodings)} faces from selected images for new search...")
            self.start_matching_process()

    def navigate_back(self):
        if self.search_history:
            self.reference_encodings = self.search_history.pop()
            if not self.search_history:
                self.btn_back.config(state="disabled")
            self.status_label.config(text="Returning to previous search...")
            self.start_matching_process()

    def update_ui_state(self):
        """Update the state of all UI elements based on current state"""
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