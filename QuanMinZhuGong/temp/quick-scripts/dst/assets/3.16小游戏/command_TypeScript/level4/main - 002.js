
                (function() {
                    var nodeEnv = typeof require !== 'undefined' && typeof process !== 'undefined';
                    var __module = nodeEnv ? module : {exports:{}};
                    var __filename = 'preview-scripts/assets/3.16小游戏/command_TypeScript/level4/main - 002.js';
                    var __require = nodeEnv ? function (request) {
                        return cc.require(request);
                    } : function (request) {
                        return __quick_compile_project__.require(request, __filename);
                    };
                    function __define (exports, require, module) {
                        if (!nodeEnv) {__quick_compile_project__.registerModule(__filename, module);}"use strict";
cc._RF.push(module, '4fdc0OSg/hIE7QH4s2Hqk/c', 'main - 002');
// 3.16小游戏/command_TypeScript/level4/main - 002.ts

"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var global_1 = require("./global");
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var leftToRight_1 = /** @class */ (function (_super) {
    __extends(leftToRight_1, _super);
    function leftToRight_1() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    // LIFE-CYCLE CALLBACKS:
    leftToRight_1.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    //产生碰撞会调用
    leftToRight_1.prototype.onCollisionEnter = function (other, self) {
        cc.log(other.node.name);
        if (global_1.default.attack == true) {
            var damage = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage.getComponent(cc.Label).string = "-30";
            var damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
            if (other.node.name == "小鬼") {
                global_1.default.minion1_hp -= 30;
            }
            else if (other.node.name == "小鬼2") {
                global_1.default.minion2_hp -= 30;
            }
            else if (other.node.name == "小鬼3") {
                global_1.default.minion3_hp -= 30;
            }
        }
        global_1.default.attack = false;
    };
    leftToRight_1.prototype.onCollisionStay = function (other) {
        global_1.default.attack = false;
        global_1.default.minion_attack = false;
    };
    leftToRight_1.prototype.onCollisionExit = function (other) {
        if ((global_1.default.minion1_hp <= 0) && (other.node.name == "小鬼")) {
            other.node.active = false;
        }
        else if ((global_1.default.minion2_hp <= 0) && (other.node.name == "小鬼2")) {
            other.node.active = false;
        }
        else if ((global_1.default.minion3_hp <= 0) && (other.node.name == "小鬼3")) {
            other.node.active = false;
            cc.director.loadScene("fight5");
        }
        else {
            if (other.node.name == "小鬼") {
                other.node.children[0].setContentSize(global_1.default.minion1_hp, 19);
            }
            else if (other.node.name == "小鬼2") {
                other.node.children[0].setContentSize(global_1.default.minion2_hp, 19);
            }
            else if (other.node.name == "小鬼3") {
                other.node.children[0].setContentSize(global_1.default.minion3_hp, 19);
            }
        }
    };
    leftToRight_1.prototype.start = function () {
        this.main_x = this.node.position.x;
    };
    leftToRight_1.prototype.update = function (dt) {
        if (global_1.default.attack == true) {
            this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x <= this.main_x + 50) && (this.node.position.x >= this.main_x - 50)) {
                this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
            }
        }
    };
    __decorate([
        property(cc.Label)
    ], leftToRight_1.prototype, "label", void 0);
    __decorate([
        property
    ], leftToRight_1.prototype, "text", void 0);
    leftToRight_1 = __decorate([
        ccclass
    ], leftToRight_1);
    return leftToRight_1;
}(cc.Component));
exports.default = leftToRight_1;

cc._RF.pop();
                    }
                    if (nodeEnv) {
                        __define(__module.exports, __require, __module);
                    }
                    else {
                        __quick_compile_project__.registerModuleFunc(__filename, function () {
                            __define(__module.exports, __require, __module);
                        });
                    }
                })();
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbImFzc2V0c1xcMy4xNuWwj+a4uOaIj1xcY29tbWFuZF9UeXBlU2NyaXB0XFxsZXZlbDRcXG1haW4gLSAwMDIudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUEsb0JBQW9CO0FBQ3BCLHdFQUF3RTtBQUN4RSxtQkFBbUI7QUFDbkIsa0ZBQWtGO0FBQ2xGLDhCQUE4QjtBQUM5QixrRkFBa0Y7QUFDbEYsbUNBQTZCO0FBS3ZCLElBQUEsS0FBc0IsRUFBRSxDQUFDLFVBQVUsRUFBbEMsT0FBTyxhQUFBLEVBQUUsUUFBUSxjQUFpQixDQUFDO0FBRzFDO0lBQTJDLGlDQUFZO0lBQXZEO1FBQUEscUVBOEZDO1FBM0ZHLFdBQUssR0FBYSxJQUFJLENBQUM7UUFHdkIsVUFBSSxHQUFXLE9BQU8sQ0FBQzs7SUF3RjNCLENBQUM7SUFsRkcsd0JBQXdCO0lBRXhCLDhCQUFNLEdBQU47UUFDSSxJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLG1CQUFtQixFQUFFLENBQUM7UUFDaEQsT0FBTyxDQUFDLE9BQU8sR0FBRyxJQUFJLENBQUM7SUFDM0IsQ0FBQztJQUdELFNBQVM7SUFDVCx3Q0FBZ0IsR0FBaEIsVUFBaUIsS0FBSyxFQUFDLElBQUk7UUFFeEIsRUFBRSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ3ZCLElBQUcsZ0JBQU0sQ0FBQyxNQUFNLElBQUksSUFBSSxFQUFDO1lBQ3JCLElBQUksTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsK0JBQStCLENBQUMsQ0FBQztZQUN0RCxNQUFNLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxNQUFNLEdBQUcsS0FBSyxDQUFDO1lBQzdDLElBQUksT0FBTyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsNkJBQTZCLENBQUMsQ0FBQztZQUNyRCxPQUFPLENBQUMs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LFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQztTQUc1RTthQUFNO1lBQ0gsSUFBSSxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFJLElBQUksQ0FBQyxNQUFNLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUksSUFBSSxDQUFDLE1BQU0sR0FBRyxFQUFFLENBQUMsRUFBSTtnQkFDakcsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFHLElBQUksR0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7YUFDM0U7U0FFSjtJQUdMLENBQUM7SUF6RkQ7UUFEQyxRQUFRLENBQUMsRUFBRSxDQUFDLEtBQUssQ0FBQztnREFDSTtJQUd2QjtRQURDLFFBQVE7K0NBQ2M7SUFOTixhQUFhO1FBRGpDLE9BQU87T0FDYSxhQUFhLENBOEZqQztJQUFELG9CQUFDO0NBOUZELEFBOEZDLENBOUYwQyxFQUFFLENBQUMsU0FBUyxHQThGdEQ7a0JBOUZvQixhQUFhIiwiZmlsZSI6IiIsInNvdXJjZVJvb3QiOiIvIiwic291cmNlc0NvbnRlbnQiOlsiLy8gTGVhcm4gVHlwZVNjcmlwdDpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvdHlwZXNjcmlwdC5odG1sXHJcbi8vIExlYXJuIEF0dHJpYnV0ZTpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvcmVmZXJlbmNlL2F0dHJpYnV0ZXMuaHRtbFxyXG4vLyBMZWFybiBsaWZlLWN5Y2xlIGNhbGxiYWNrczpcclxuLy8gIC0gaHR0cHM6Ly9kb2NzLmNvY29zLmNvbS9jcmVhdG9yL21hbnVhbC9lbi9zY3JpcHRpbmcvbGlmZS1jeWNsZS1jYWxsYmFja3MuaHRtbFxyXG5pbXBvcnQgZ2xvYmFsIGZyb20gXCIuL2dsb2JhbFwiXHJcblxyXG5cclxuXHJcblxyXG5jb25zdCB7Y2NjbGFzcywgcHJvcGVydHl9ID0gY2MuX2RlY29yYXRvcjtcclxuXHJcbkBjY2NsYXNzXHJcbmV4cG9ydCBkZWZhdWx0IGNsYXNzIGxlZnRUb1JpZ2h0XzEgZXh0ZW5kcyBjYy5Db21wb25lbnQge1xyXG5cclxuICAgIEBwcm9wZXJ0eShjYy5MYWJlbClcclxuICAgIGxhYmVsOiBjYy5MYWJlbCA9IG51bGw7XHJcblxyXG4gICAgQHByb3BlcnR5XHJcbiAgICB0ZXh0OiBzdHJpbmcgPSAnaGVsbG8nO1xyXG4gICAgbWFpbl94OiBudW1iZXI7XHJcbiAgICBcclxuICAgIHB1YmxpYyBzdGF0aWMgY3VycmVudF94OiBudW1iZXI7XHJcbiAgICBwdWJsaWMgc3RhdGljIGN1cnJlbnRfeTogbnVtYmVyO1xyXG5cclxuICAgIC8vIExJRkUtQ1lDTEUgQ0FMTEJBQ0tTOlxyXG5cclxuICAgIG9uTG9hZCAoKSB7XHJcbiAgICAgICAgdmFyIG1hbmFnZXIgPSBjYy5kaXJlY3Rvci5nZXRDb2xsaXNpb25NYW5hZ2VyKCk7XHJcbiAgICAgICAgbWFuYWdlci5lbmFibGVkID0gdHJ1ZTtcclxuICAgIH1cclxuICAgIFxyXG5cclxuICAgIC8v5Lqn55Sf56Kw5pKe5Lya6LCD55SoXHJcbiAgICBvbkNvbGxpc2lvbkVudGVyKG90aGVyLHNlbGYpe1xyXG4gICAgICBcclxuICAgICAgIGNjLmxvZyhvdGhlci5ub2RlLm5hbWUpO1xyXG4gICAgICAgIGlmKGdsb2JhbC5hdHRhY2sgPT0gdHJ1ZSl7XHJcbiAgICAgICAgICAgIGxldCBkYW1hZ2UgPSBjYy5maW5kKFwiQ2FudmFzL2JqL2t1YW4veHkvbWFpbl9kYW1hZ2VcIik7XHJcbiAgICAgICAgICAgIGRhbWFnZS5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiLTMwXCI7XHJcbiAgICAgICAgICAgIGxldCBkYW1hZ2UyID0gY2MuZmluZChcIkNhbnZhcy9iai9rdWFuL2VuZW15X2RhbWFnZVwiKTtcclxuICAgICAgICAgICAgZGFtYWdlMi5nZXRDb21wb25lbnQoY2MuTGFiZWwpLnN0cmluZyA9IFwiXCI7XHJcbiAgICAgICAgICAgaWYob3RoZXIubm9kZS5uYW1lID09IFwi5bCP6ay8XCIpIHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1pbmlvbjFfaHAgLT0gMzA7XHJcbiAgICAgICAgICAgfSBlbHNlICBpZihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLwyXCIpIHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1pbmlvbjJfaHAgLT0gMzA7XHJcbiAgICAgICAgICAgfSBlbHNlICBpZihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLwzXCIpIHtcclxuICAgICAgICAgICAgZ2xvYmFsLm1pbmlvbjNfaHAgLT0gMzA7XHJcbiAgICAgICAgICAgfVxyXG4gICAgICAgIH1cclxuICAgICAgICBcclxuICAgICAgICBnbG9iYWwuYXR0YWNrID0gZmFsc2U7XHJcbiAgICB9XHJcblxyXG5cclxuICAgIG9uQ29sbGlzaW9uU3RheShvdGhlcikge1xyXG4gICAgICAgIGdsb2JhbC5hdHRhY2sgPSBmYWxzZTtcclxuICAgICAgICBnbG9iYWwubWluaW9uX2F0dGFjayA9IGZhbHNlO1xyXG4gICAgfVxyXG5cclxuICAgIG9uQ29sbGlzaW9uRXhpdChvdGhlcikge1xyXG5cclxuICAgICAgIFxyXG4gICAgICAgXHJcbiAgICAgXHJcbiAgICAgIFxyXG4gICAgICAgaWYoKGdsb2JhbC5taW5pb24xX2hwIDw9IDApJiYob3RoZXIubm9kZS5uYW1lID09IFwi5bCP6ay8XCIpKSB7ICAgXHJcbiAgICAgICAgb3RoZXIubm9kZS5hY3RpdmUgPSBmYWxzZTtcclxuICAgICAgIH0gIGVsc2UgaWYoKGdsb2JhbC5taW5pb24yX2hwIDw9IDApJiYob3RoZXIubm9kZS5uYW1lID09IFwi5bCP6ay8MlwiKSkgeyAgIFxyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICB9IGVsc2UgaWYoKGdsb2JhbC5taW5pb24zX2hwIDw9IDApJiYob3RoZXIubm9kZS5uYW1lID09IFwi5bCP6ay8M1wiKSkgeyAgIFxyXG4gICAgICAgIG90aGVyLm5vZGUuYWN0aXZlID0gZmFsc2U7XHJcbiAgICAgICAgY2MuZGlyZWN0b3IubG9hZFNjZW5lKFwiZmlnaHQ1XCIpO1xyXG4gICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgICBpZihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLxcIikge1xyXG4gICAgICAgICAgICBvdGhlci5ub2RlLmNoaWxkcmVuWzBdLnNldENvbnRlbnRTaXplKGdsb2JhbC5taW5pb24xX2hwLCAxOSk7XHJcbiAgICAgICAgICAgfSBlbHNlICBpZihvdGhlci5ub2RlLm5hbWUgPT0gXCLlsI/prLwyXCIpIHtcclxuICAgICAgICAgICAgb3RoZXIubm9kZS5jaGlsZHJlblswXS5zZXRDb250ZW50U2l6ZShnbG9iYWwubWluaW9uMl9ocCwgMTkpO1xyXG4gICAgICAgICAgIH0gZWxzZSAgaWYob3RoZXIubm9kZS5uYW1lID09IFwi5bCP6ay8M1wiKSB7XHJcbiAgICAgICAgICAgIG90aGVyLm5vZGUuY2hpbGRyZW5bMF0uc2V0Q29udGVudFNpemUoZ2xvYmFsLm1pbmlvbjNfaHAsIDE5KTtcclxuICAgICAgICAgICB9XHJcbiAgICAgICB9XHJcbiAgICAgICBcclxuICAgIH1cclxuICAgIFxyXG4gICAgc3RhcnQgKCkge1xyXG4gICAgICBcclxuICAgIHRoaXMubWFpbl94ID0gdGhpcy5ub2RlLnBvc2l0aW9uLng7XHJcbiAgICAgICBcclxuICAgICBcclxuICAgIH1cclxuXHJcbiAgICB1cGRhdGUgKGR0KSB7XHJcbiAgICAgICAgaWYgKGdsb2JhbC5hdHRhY2sgPT0gIHRydWUpIHtcclxuICAgICAgICAgdGhpcy5ub2RlLnNldFBvc2l0aW9uKHRoaXMubm9kZS5wb3NpdGlvbi54ICsgMTAwMCpkdCwgdGhpcy5ub2RlLnBvc2l0aW9uLnkpO1xyXG4gICAgICAgICAgICBcclxuICAgICAgICBcclxuICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgICBpZiAoISh0aGlzLm5vZGUucG9zaXRpb24ueCA8PSB0aGlzLm1haW5feCArIDUwKSAmJiAodGhpcy5ub2RlLnBvc2l0aW9uLnggPj0gdGhpcy5tYWluX3ggLSA1MCkpICAge1xyXG4gICAgICAgICAgICB0aGlzLm5vZGUuc2V0UG9zaXRpb24odGhpcy5ub2RlLnBvc2l0aW9uLnggLSAxMDAwKmR0LCB0aGlzLm5vZGUucG9zaXRpb24ueSk7XHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgXHJcbiAgICAgICAgfVxyXG5cclxuICAgICBcclxuICAgIH1cclxuICAgIFxyXG59XHJcblxyXG5cclxuIl19